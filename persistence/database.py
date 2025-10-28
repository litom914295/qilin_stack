"""
数据库持久化 - PostgreSQL/SQLite 双后端
"""
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import json
import os


@dataclass
class DecisionRecord:
    """决策记录"""
    id: Optional[int] = None
    timestamp: datetime = None
    symbol: str = ""
    signal: str = ""
    confidence: float = 0.0
    strength: float = 0.0
    reasoning: str = ""
    source_signals: str = ""  # JSON
    market_state: str = ""


@dataclass
class PerformanceRecord:
    """性能记录"""
    id: Optional[int] = None
    timestamp: datetime = None
    system: str = ""
    accuracy: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    sample_size: int = 0


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or self._default_connection()
        self.conn = None
        self._is_pg = False
        self._connect()
    
    def _default_connection(self) -> str:
        """默认连接字符串（可由ENV覆盖）"""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        db = os.getenv('DB_NAME', 'qilin_stack')
        user = os.getenv('DB_USER', 'admin')
        password = os.getenv('DB_PASSWORD', 'changeme')
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def _connect(self):
        """连接数据库，优先Postgres，失败回退SQLite"""
        try:
            import psycopg2  # type: ignore
            self.conn = psycopg2.connect(self.connection_string)
            self._is_pg = True
            self._init_tables()
        except Exception:
            # 回退到SQLite
            import sqlite3
            self.conn = sqlite3.connect(os.getenv('SQLITE_PATH', 'qilin_stack.db'))
            self._is_pg = False
            self._init_tables()
    
    def _adapt_sql(self, sql: str) -> str:
        """将统一的'?'占位符转为目标驱动风格"""
        if self._is_pg:
            return sql.replace('?', '%s')
        return sql
    
    def _init_tables(self):
        """初始化表（按不同后端DDL）"""
        cursor = self.conn.cursor()
        if self._is_pg:
            id_def = 'SERIAL PRIMARY KEY'
        else:
            id_def = 'INTEGER PRIMARY KEY AUTOINCREMENT'
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS decisions (
                id {id_def},
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                signal VARCHAR(20) NOT NULL,
                confidence FLOAT NOT NULL,
                strength FLOAT NOT NULL,
                reasoning TEXT,
                source_signals TEXT,
                market_state VARCHAR(20)
            )
        """)
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS performance (
                id {id_def},
                timestamp TIMESTAMP NOT NULL,
                system VARCHAR(50) NOT NULL,
                accuracy FLOAT,
                f1_score FLOAT,
                sharpe_ratio FLOAT,
                win_rate FLOAT,
                sample_size INTEGER
            )
        """)
        
        # 索引
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance(timestamp)")
        except Exception:
            pass
        
        self.conn.commit()
    
    def save_decision(self, decision: DecisionRecord) -> int:
        """保存决策，返回ID"""
        cursor = self.conn.cursor()
        params = (
            decision.timestamp or datetime.now(),
            decision.symbol,
            decision.signal,
            decision.confidence,
            decision.strength,
            decision.reasoning,
            decision.source_signals,
            decision.market_state,
        )
        if self._is_pg:
            sql = (
                "INSERT INTO decisions (timestamp, symbol, signal, confidence, strength, reasoning, source_signals, market_state) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id"
            )
            cursor.execute(sql, params)
            rid = cursor.fetchone()[0]
        else:
            sql = self._adapt_sql(
                "INSERT INTO decisions (timestamp, symbol, signal, confidence, strength, reasoning, source_signals, market_state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            cursor.execute(sql, params)
            rid = cursor.lastrowid
        self.conn.commit()
        return int(rid)
    
    def get_decisions(self, symbol: Optional[str] = None, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 100) -> List[DecisionRecord]:
        """查询决策"""
        cursor = self.conn.cursor()
        query = "SELECT * FROM decisions WHERE 1=1"
        params: list = []
        if symbol:
            query += " AND symbol = ?"; params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"; params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"; params.append(end_date)
        query += f" ORDER BY timestamp DESC LIMIT {int(limit)}"
        cursor.execute(self._adapt_sql(query), params)
        rows = cursor.fetchall()
        return [DecisionRecord(*row) for row in rows]
    
    def save_performance(self, performance: PerformanceRecord) -> int:
        """保存性能记录，返回ID"""
        cursor = self.conn.cursor()
        params = (
            performance.timestamp or datetime.now(),
            performance.system,
            performance.accuracy,
            performance.f1_score,
            performance.sharpe_ratio,
            performance.win_rate,
            performance.sample_size,
        )
        if self._is_pg:
            sql = (
                "INSERT INTO performance (timestamp, system, accuracy, f1_score, sharpe_ratio, win_rate, sample_size) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id"
            )
            cursor.execute(sql, params)
            rid = cursor.fetchone()[0]
        else:
            sql = self._adapt_sql(
                "INSERT INTO performance (timestamp, system, accuracy, f1_score, sharpe_ratio, win_rate, sample_size) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            cursor.execute(sql, params)
            rid = cursor.lastrowid
        self.conn.commit()
        return int(rid)
    
    def get_performance_stats(self, system: str, days: int = 30) -> Dict:
        """获取近N天性能均值"""
        cursor = self.conn.cursor()
        if self._is_pg:
            sql = (
                "SELECT AVG(accuracy), AVG(f1_score), AVG(sharpe_ratio), AVG(win_rate) "
                "FROM performance WHERE system = %s AND timestamp >= NOW() - INTERVAL %s"
            )
            cursor.execute(sql, (system, f"{int(days)} days"))
        else:
            sql = self._adapt_sql(
                "SELECT AVG(accuracy), AVG(f1_score), AVG(sharpe_ratio), AVG(win_rate) "
                "FROM performance WHERE system = ? AND timestamp >= datetime('now', ?)"
            )
            cursor.execute(sql, (system, f"-{int(days)} days"))
        row = cursor.fetchone()
        if row:
            return {
                'avg_accuracy': row[0] or 0,
                'avg_f1_score': row[1] or 0,
                'avg_sharpe_ratio': row[2] or 0,
                'avg_win_rate': row[3] or 0,
            }
        return {}
    
    def archive_old_data(self, days: int = 90):
        """删除N天前数据"""
        cursor = self.conn.cursor()
        if self._is_pg:
            sql = "DELETE FROM decisions WHERE timestamp < NOW() - INTERVAL %s"
            cursor.execute(sql, (f"{int(days)} days",))
        else:
            sql = self._adapt_sql("DELETE FROM decisions WHERE timestamp < datetime('now', ?)")
            cursor.execute(sql, (f"-{int(days)} days",))
        deleted = cursor.rowcount
        self.conn.commit()
        return int(deleted)
    
    def close(self):
        if self.conn:
            self.conn.close()
    

# 全局实例
_db = None

def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db
