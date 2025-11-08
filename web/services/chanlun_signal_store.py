"""缠论信号存储服务 - P2-2

目标:
- 统一管理缠论信号的本地持久化(默认SQLite)
- 提供写入/读取/统计接口,便于Dashboard与回测共享

用法:

from web.services.chanlun_signal_store import ChanLunSignalStore

store = ChanLunSignalStore(db_path='data/chanlun_signals.sqlite')
store.init()

# 写入
store.save_signals(df)  # df列: ['time','symbol','signal_type','price','score','status']

# 读取
recent = store.load_signals(limit=200)

# 统计
stats = store.get_daily_stats()

"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


DDL = """
CREATE TABLE IF NOT EXISTS chanlun_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    price REAL NOT NULL,
    score REAL DEFAULT 0,
    status TEXT DEFAULT '待确认'
);
CREATE INDEX IF NOT EXISTS idx_signals_time ON chanlun_signals(time);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON chanlun_signals(symbol);
"""


@dataclass
class ChanLunSignalStore:
    db_path: str = 'data/chanlun_signals.sqlite'

    def init(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(DDL)

    def save_signals(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        cols = ['time','symbol','signal_type','price','score','status']
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f'missing columns: {missing}')
        with sqlite3.connect(self.db_path) as conn:
            df[cols].to_sql('chanlun_signals', conn, if_exists='append', index=False)
            cur = conn.execute('SELECT changes()')
            return cur.fetchone()[0] or 0

    def load_signals(self, limit: int = 500, symbol: Optional[str] = None) -> pd.DataFrame:
        sql = 'SELECT time, symbol, signal_type, price, score, status FROM chanlun_signals'
        params: Dict[str, Any] = {}
        if symbol:
            sql += ' WHERE symbol = :symbol'
            params['symbol'] = symbol
        sql += ' ORDER BY time DESC LIMIT :limit'
        params['limit'] = int(limit)
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_daily_stats(self) -> pd.DataFrame:
        sql = (
            "SELECT substr(time,1,10) AS day, "
            "COUNT(*) AS cnt, "
            "SUM(CASE WHEN signal_type LIKE '%买' THEN 1 ELSE 0 END) AS buy_cnt, "
            "SUM(CASE WHEN signal_type LIKE '%卖' THEN 1 ELSE 0 END) AS sell_cnt, "
            "AVG(score) AS avg_score "
            "FROM chanlun_signals GROUP BY day ORDER BY day DESC"
        )
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn)
