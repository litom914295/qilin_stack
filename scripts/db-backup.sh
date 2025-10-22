#!/bin/bash
# 数据库备份脚本
# RPO目标: <5分钟
# RTO目标: <15分钟

set -e

# 配置
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-qilin_trading}"
DB_USER="${DB_USER:-postgres}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/qilin}"
S3_BUCKET="${S3_BUCKET:-s3://qilin-backups}"
RETENTION_DAYS=30

# 日志
LOG_FILE="/var/log/qilin/db-backup.log"
mkdir -p "$(dirname $LOG_FILE)"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $1"
    exit 1
}

# 检查必要工具
check_dependencies() {
    local deps=("pg_dump" "pg_dumpall" "aws" "gzip")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "$dep is not installed"
        fi
    done
}

# 创建备份目录
ensure_backup_dir() {
    mkdir -p "$BACKUP_DIR"/{full,incremental,wal}
    log "Backup directory: $BACKUP_DIR"
}

# 全量备份
full_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/full/qilin_${timestamp}_full.sql.gz"
    
    log "Starting full backup..."
    
    # 备份数据库
    PGPASSWORD="$DB_PASSWORD" pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --format=custom \
        --compress=9 \
        --verbose \
        --file="${backup_file%.gz}" 2>&1 | tee -a "$LOG_FILE"
    
    # 压缩
    gzip "${backup_file%.gz}"
    
    # 生成校验和
    sha256sum "$backup_file" > "${backup_file}.sha256"
    
    # 备份全局对象（角色、表空间等）
    local globals_file="$BACKUP_DIR/full/qilin_${timestamp}_globals.sql.gz"
    PGPASSWORD="$DB_PASSWORD" pg_dumpall \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        --globals-only | gzip > "$globals_file"
    
    log "Full backup completed: $backup_file"
    log "Backup size: $(du -h $backup_file | cut -f1)"
    
    # 上传到S3
    upload_to_s3 "$backup_file" "full/"
    upload_to_s3 "${backup_file}.sha256" "full/"
    upload_to_s3 "$globals_file" "full/"
    
    # 记录备份元数据
    echo "{
        \"type\": \"full\",
        \"timestamp\": \"$timestamp\",
        \"file\": \"$backup_file\",
        \"size_bytes\": $(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file"),
        \"checksum\": \"$(cat ${backup_file}.sha256 | cut -d' ' -f1)\"
    }" > "$BACKUP_DIR/full/backup_${timestamp}.json"
    
    echo "$backup_file"
}

# 增量备份（基于WAL归档）
incremental_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local wal_backup_dir="$BACKUP_DIR/wal/$timestamp"
    
    log "Starting incremental backup (WAL archiving)..."
    
    mkdir -p "$wal_backup_dir"
    
    # 归档WAL文件
    # 注意：需要在PostgreSQL配置中启用归档模式
    # archive_mode = on
    # archive_command = 'cp %p /var/backups/qilin/wal/%f'
    
    # 查找最新WAL文件
    local latest_wal=$(PGPASSWORD="$DB_PASSWORD" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -t -c "SELECT pg_walfile_name(pg_current_wal_lsn())")
    
    log "Latest WAL file: $latest_wal"
    
    # 复制WAL文件到备份目录
    local pg_wal_dir="${PGDATA:-/var/lib/postgresql/data}/pg_wal"
    if [ -d "$pg_wal_dir" ]; then
        rsync -av "$pg_wal_dir/" "$wal_backup_dir/" 2>&1 | tee -a "$LOG_FILE"
    fi
    
    # 压缩WAL归档
    tar -czf "$BACKUP_DIR/incremental/wal_${timestamp}.tar.gz" \
        -C "$BACKUP_DIR/wal" "$timestamp"
    
    log "Incremental backup completed"
    
    # 上传到S3
    upload_to_s3 "$BACKUP_DIR/incremental/wal_${timestamp}.tar.gz" "incremental/"
}

# 上传到S3
upload_to_s3() {
    local file="$1"
    local prefix="$2"
    
    if [ -z "$S3_BUCKET" ]; then
        log "S3_BUCKET not configured, skipping upload"
        return
    fi
    
    log "Uploading to S3: $file"
    aws s3 cp "$file" "${S3_BUCKET}/${prefix}$(basename $file)" \
        --storage-class STANDARD_IA \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "Upload successful"
    else
        error "Upload failed"
    fi
}

# 验证备份
verify_backup() {
    local backup_file="$1"
    
    log "Verifying backup: $backup_file"
    
    # 检查文件存在
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
    fi
    
    # 验证校验和
    if [ -f "${backup_file}.sha256" ]; then
        if sha256sum -c "${backup_file}.sha256"; then
            log "Checksum verification passed"
        else
            error "Checksum verification failed"
        fi
    fi
    
    # 测试解压
    log "Testing decompression..."
    if zcat "$backup_file" > /dev/null 2>&1; then
        log "Decompression test passed"
    else
        error "Decompression test failed"
    fi
    
    # 测试pg_restore（不实际恢复）
    log "Testing pg_restore..."
    if pg_restore --list "${backup_file%.gz}" > /dev/null 2>&1; then
        log "pg_restore test passed"
    else
        error "pg_restore test failed"
    fi
    
    log "Backup verification completed successfully"
}

# 清理旧备份
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days..."
    
    # 本地清理
    find "$BACKUP_DIR/full" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/incremental" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    
    # S3清理（使用生命周期策略更好）
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)
    
    log "Cleanup completed"
}

# 获取备份列表
list_backups() {
    log "Available backups:"
    echo ""
    echo "=== Full Backups ==="
    ls -lh "$BACKUP_DIR/full/"*.sql.gz 2>/dev/null || echo "No full backups found"
    echo ""
    echo "=== Incremental Backups ==="
    ls -lh "$BACKUP_DIR/incremental/"*.tar.gz 2>/dev/null || echo "No incremental backups found"
}

# 主函数
main() {
    local backup_type="${1:-full}"
    local verify_only="${2:-false}"
    
    log "=== Qilin Database Backup Started ==="
    log "Type: $backup_type"
    
    check_dependencies
    ensure_backup_dir
    
    case "$backup_type" in
        full)
            backup_file=$(full_backup)
            if [ "$verify_only" != "--skip-verify" ]; then
                verify_backup "$backup_file"
            fi
            ;;
        incremental)
            incremental_backup
            ;;
        verify)
            if [ -z "$3" ]; then
                error "Please specify backup file to verify"
            fi
            verify_backup "$3"
            ;;
        list)
            list_backups
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        *)
            echo "Usage: $0 {full|incremental|verify <file>|list|cleanup}"
            exit 1
            ;;
    esac
    
    log "=== Backup Completed Successfully ==="
}

# 执行
main "$@"
