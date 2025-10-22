"""
数据库持久化 - PostgreSQL
"""
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json


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
        self._connect()
    
    def _default_connection(self) -> str:
        """默认连接字符串"""
        import os
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        db = os.getenv('DB_NAME', 'qilin_stack')
        user = os.getenv('DB_USER', 'admin')
        password = os.getenv('DB_PASSWORD', 'changeme')
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def _connect(self):
        """连接数据库"""
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.connection_string)
            self._init_tables()
        except ImportError:
            print("⚠️ psycopg2未安装，使用SQLite")
            self._use_sqlite()
        except Exception as e:
            print(f"⚠️ 数据库连接失败: {e}，使用SQLite")
            self._use_sqlite()
    
    def _use_sqlite(self):
        """使用SQLite作为备选"""
        import sqlite3
        self.conn = sqlite3.connect('qilin_stack.db')
        self._init_tables()
    
    def _init_tables(self):
        """初始化表"""
        cursor = self.conn.cursor()
        
        # 决策表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id SERIAL PRIMARY KEY,
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
        
        # 性能表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                system VARCHAR(50) NOT NULL,
                accuracy FLOAT,
                f1_score FLOAT,
                sharpe_ratio FLOAT,
                win_rate FLOAT,
                sample_size INTEGER
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance(timestamp)")
        
        self.conn.commit()
    
    def save_decision(self, decision: DecisionRecord) -> int:
        """保存决策"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO decisions (timestamp, symbol, signal, confidence, strength, 
                                 reasoning, source_signals, market_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.timestamp or datetime.now(),
            decision.symbol,
            decision.signal,
            decision.confidence,
            decision.strength,
            decision.reasoning,
            decision.source_signals,
            decision.market_state
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_decisions(self, symbol: Optional[str] = None, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 100) -> List[DecisionRecord]:
        """查询决策"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM decisions WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [DecisionRecord(*row) for row in rows]
    
    def save_performance(self, performance: PerformanceRecord) -> int:
        """保存性能记录"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO performance (timestamp, system, accuracy, f1_score, 
                                   sharpe_ratio, win_rate, sample_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            performance.timestamp or datetime.now(),
            performance.system,
            performance.accuracy,
            performance.f1_score,
            performance.sharpe_ratio,
            performance.win_rate,
            performance.sample_size
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_performance_stats(self, system: str, days: int = 30) -> Dict:
        """获取性能统计"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT AVG(accuracy), AVG(f1_score), AVG(sharpe_ratio), AVG(win_rate)
            FROM performance
            WHERE system = ? AND timestamp >= datetime('now', ?)
        """, (system, f'-{days} days'))
        
        row = cursor.fetchone()
        if row:
            return {
                'avg_accuracy': row[0] or 0,
                'avg_f1_score': row[1] or 0,
                'avg_sharpe_ratio': row[2] or 0,
                'avg_win_rate': row[3] or 0
            }
        return {}
    
    def archive_old_data(self, days: int = 90):
        """归档旧数据"""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM decisions 
            WHERE timestamp < datetime('now', ?)
        """, (f'-{days} days',))
        
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()


# 全局数据库实例
_db = None

def get_db() -> DatabaseManager:
    """获取全局数据库实例"""
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db
