"""
批量数据库操作优化器 (Batch Database Operations)
提供高性能的批量插入、更新和查询功能
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("⚠️  SQLAlchemy未安装，批量操作功能将受限。安装方法: pip install sqlalchemy")

logger = logging.getLogger(__name__)


class BatchDatabaseOperations:
    """
    批量数据库操作类
    
    特性:
    1. 批量插入优化（使用bulk_insert_mappings）
    2. 批量更新优化
    3. 事务管理
    4. 连接池管理
    5. 自动批次分割
    """
    
    def __init__(
        self,
        database_url: str,
        batch_size: int = 1000,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        """
        初始化批量操作器
        
        Args:
            database_url: 数据库连接URL
            batch_size: 每批次操作的记录数
            pool_size: 连接池大小
            max_overflow: 连接池最大溢出数
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("需要安装SQLAlchemy: pip install sqlalchemy")
        
        self.database_url = database_url
        self.batch_size = batch_size
        
        # 创建引擎（使用连接池）
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # 自动ping检查连接有效性
            echo=False
        )
        
        # 创建Session工厂
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"✅ 批量数据库操作器已初始化: batch_size={batch_size}")
    
    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话（上下文管理器）"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()
    
    def bulk_insert(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        conflict_action: str = 'ignore'
    ) -> int:
        """
        批量插入数据
        
        Args:
            table_name: 表名
            data: 数据列表，每项是一个字典
            conflict_action: 冲突处理方式 ('ignore', 'update')
            
        Returns:
            插入的记录数
        """
        if not data:
            return 0
        
        total_inserted = 0
        
        try:
            # 分批处理
            batches = self._split_into_batches(data)
            
            with self.get_session() as session:
                for batch in batches:
                    try:
                        if conflict_action == 'ignore':
                            # 使用INSERT IGNORE (MySQL) 或 ON CONFLICT DO NOTHING (PostgreSQL)
                            self._bulk_insert_ignore(session, table_name, batch)
                        elif conflict_action == 'update':
                            # 使用 ON DUPLICATE KEY UPDATE (MySQL) 或 ON CONFLICT DO UPDATE (PostgreSQL)
                            self._bulk_insert_update(session, table_name, batch)
                        else:
                            # 普通插入
                            self._bulk_insert_normal(session, table_name, batch)
                        
                        total_inserted += len(batch)
                        
                    except Exception as e:
                        logger.error(f"批次插入失败: {e}")
                        raise
            
            logger.info(f"✅ 批量插入完成: {total_inserted} 条记录")
            return total_inserted
            
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            raise
    
    def _bulk_insert_normal(
        self,
        session: Session,
        table_name: str,
        data: List[Dict[str, Any]]
    ):
        """普通批量插入"""
        # 构建INSERT语句
        if not data:
            return
        
        columns = list(data[0].keys())
        placeholders = ', '.join([f':{col}' for col in columns])
        columns_str = ', '.join(columns)
        
        sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        
        session.execute(text(sql), data)
    
    def _bulk_insert_ignore(
        self,
        session: Session,
        table_name: str,
        data: List[Dict[str, Any]]
    ):
        """批量插入（忽略冲突）"""
        if not data:
            return
        
        columns = list(data[0].keys())
        placeholders = ', '.join([f':{col}' for col in columns])
        columns_str = ', '.join(columns)
        
        # 根据数据库类型使用不同的语法
        if 'postgresql' in self.database_url:
            sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT DO NOTHING
            """
        elif 'mysql' in self.database_url:
            sql = f"""
                INSERT IGNORE INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """
        else:
            # SQLite
            sql = f"""
                INSERT OR IGNORE INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """
        
        session.execute(text(sql), data)
    
    def _bulk_insert_update(
        self,
        session: Session,
        table_name: str,
        data: List[Dict[str, Any]]
    ):
        """批量插入或更新"""
        if not data:
            return
        
        columns = list(data[0].keys())
        placeholders = ', '.join([f':{col}' for col in columns])
        columns_str = ', '.join(columns)
        
        # 假设第一列是主键
        pk_column = columns[0]
        update_columns = columns[1:]
        update_str = ', '.join([f"{col} = excluded.{col}" for col in update_columns])
        
        if 'postgresql' in self.database_url:
            sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT ({pk_column}) DO UPDATE SET {update_str}
            """
        elif 'mysql' in self.database_url:
            update_str = ', '.join([f"{col} = VALUES({col})" for col in update_columns])
            sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_str}
            """
        else:
            # SQLite
            sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT({pk_column}) DO UPDATE SET {update_str}
            """
        
        session.execute(text(sql), data)
    
    def bulk_update(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        key_column: str = 'id'
    ) -> int:
        """
        批量更新数据
        
        Args:
            table_name: 表名
            data: 数据列表
            key_column: 主键列名
            
        Returns:
            更新的记录数
        """
        if not data:
            return 0
        
        total_updated = 0
        
        try:
            batches = self._split_into_batches(data)
            
            with self.get_session() as session:
                for batch in batches:
                    # 构建批量UPDATE语句
                    for item in batch:
                        if key_column not in item:
                            continue
                        
                        key_value = item[key_column]
                        update_fields = {k: v for k, v in item.items() if k != key_column}
                        
                        if not update_fields:
                            continue
                        
                        set_clause = ', '.join([f"{k} = :{k}" for k in update_fields.keys()])
                        sql = f"UPDATE {table_name} SET {set_clause} WHERE {key_column} = :key_value"
                        
                        params = {**update_fields, 'key_value': key_value}
                        session.execute(text(sql), params)
                        total_updated += 1
            
            logger.info(f"✅ 批量更新完成: {total_updated} 条记录")
            return total_updated
            
        except Exception as e:
            logger.error(f"批量更新失败: {e}")
            raise
    
    def bulk_delete(
        self,
        table_name: str,
        ids: List[Any],
        key_column: str = 'id'
    ) -> int:
        """
        批量删除数据
        
        Args:
            table_name: 表名
            ids: ID列表
            key_column: 主键列名
            
        Returns:
            删除的记录数
        """
        if not ids:
            return 0
        
        try:
            with self.get_session() as session:
                # 使用IN子句批量删除
                sql = f"DELETE FROM {table_name} WHERE {key_column} IN :ids"
                result = session.execute(text(sql), {'ids': tuple(ids)})
                deleted_count = result.rowcount
            
            logger.info(f"✅ 批量删除完成: {deleted_count} 条记录")
            return deleted_count
            
        except Exception as e:
            logger.error(f"批量删除失败: {e}")
            raise
    
    def bulk_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        return_df: bool = False
    ) -> List[Dict] | pd.DataFrame:
        """
        批量查询
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            return_df: 是否返回DataFrame
            
        Returns:
            查询结果
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(sql), params or {})
                
                if return_df:
                    # 返回DataFrame
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return df
                else:
                    # 返回字典列表
                    rows = []
                    for row in result:
                        rows.append(dict(row._mapping))
                    return rows
                    
        except Exception as e:
            logger.error(f"批量查询失败: {e}")
            raise
    
    def bulk_upsert_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'append'
    ) -> int:
        """
        从DataFrame批量插入/更新数据
        
        Args:
            df: DataFrame
            table_name: 表名
            if_exists: 'append', 'replace', 'fail'
            
        Returns:
            操作的记录数
        """
        if df.empty:
            return 0
        
        try:
            # 使用pandas的to_sql方法（高度优化）
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',  # 使用多值INSERT
                chunksize=self.batch_size
            )
            
            row_count = len(df)
            logger.info(f"✅ DataFrame批量操作完成: {row_count} 条记录")
            return row_count
            
        except Exception as e:
            logger.error(f"DataFrame批量操作失败: {e}")
            raise
    
    def _split_into_batches(
        self,
        data: List[Any]
    ) -> List[List[Any]]:
        """将数据分割成批次"""
        batches = []
        for i in range(0, len(data), self.batch_size):
            batches.append(data[i:i + self.batch_size])
        return batches
    
    def execute_transaction(
        self,
        operations: List[Tuple[str, List[Dict[str, Any]]]]
    ) -> bool:
        """
        执行事务（多个操作原子性）
        
        Args:
            operations: 操作列表 [(table_name, data), ...]
            
        Returns:
            是否成功
        """
        try:
            with self.get_session() as session:
                for table_name, data in operations:
                    self._bulk_insert_normal(session, table_name, data)
                
                # 事务会在上下文退出时自动提交
            
            logger.info(f"✅ 事务执行成功: {len(operations)} 个操作")
            return True
            
        except Exception as e:
            logger.error(f"事务执行失败: {e}")
            return False
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """获取表统计信息"""
        try:
            sql = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.bulk_query(sql)
            
            return {
                'table_name': table_name,
                'row_count': result[0]['count'] if result else 0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"获取表统计信息失败: {e}")
            return {'table_name': table_name, 'error': str(e)}
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
        logger.info("数据库连接已关闭")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建批量操作器（使用SQLite作为示例）
    db_ops = BatchDatabaseOperations(
        database_url="sqlite:///./test_batch.db",
        batch_size=500
    )
    
    print("\n📊 测试批量数据库操作\n")
    
    # 创建测试表
    with db_ops.get_session() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS test_stocks (
                symbol VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100),
                price FLOAT,
                volume INTEGER,
                updated_at TIMESTAMP
            )
        """))
    
    # 准备测试数据
    test_data = [
        {
            'symbol': f'00000{i}.SZ',
            'name': f'Stock {i}',
            'price': 10.0 + i * 0.1,
            'volume': 1000 * i,
            'updated_at': datetime.now()
        }
        for i in range(1, 1001)
    ]
    
    print(f"准备插入 {len(test_data)} 条记录...")
    
    # 测试批量插入
    import time
    start = time.time()
    inserted = db_ops.bulk_insert('test_stocks', test_data, conflict_action='ignore')
    elapsed = time.time() - start
    
    print(f"✅ 插入完成: {inserted} 条记录，耗时: {elapsed:.2f}秒")
    print(f"   性能: {inserted/elapsed:.0f} 条/秒")
    
    # 查询统计
    stats = db_ops.get_table_stats('test_stocks')
    print(f"\n📈 表统计: {stats}")
    
    # 清理
    print("\n🧹 清理测试数据...")
    db_ops.close()
    import os
    if os.path.exists('test_batch.db'):
        os.remove('test_batch.db')
    print("✅ 完成")
