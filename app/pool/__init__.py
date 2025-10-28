"""
Simple in-process database pool for tests (SQLite only).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from typing import Dict, Optional, Iterator
import threading


class DatabaseType(str, Enum):
    SQLITE = "sqlite"


@dataclass
class DatabaseConfig:
    db_type: DatabaseType
    database: str
    min_size: int = 1
    max_size: int = 5


class _SQLitePool:
    def __init__(self, cfg: DatabaseConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._conns: list[sqlite3.Connection] = []
        self.current_size = 0
        self.pool_available = 0
        self.pool_in_use = 0
        self.peak_size = 0
        self.total_created = 0
        self.total_closed = 0
        self.wait_count = 0
        self.timeout_count = 0

    def _create_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cfg.database)
        self.current_size += 1
        self.total_created += 1
        self.peak_size = max(self.peak_size, self.current_size)
        return conn

    @contextmanager
    def acquire(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn: Optional[sqlite3.Connection] = None
            if self._conns:
                conn = self._conns.pop()
            elif self.current_size < self.cfg.max_size:
                conn = self._create_conn()
            else:
                # simplistic: no waiting logic, count timeout
                self.timeout_count += 1
                raise RuntimeError("Pool exhausted")
            self.pool_in_use += 1
            self.pool_available = max(0, self.current_size - self.pool_in_use)
        try:
            assert conn is not None
            yield conn
        finally:
            with self._lock:
                self.pool_in_use -= 1
                self.pool_available = max(0, self.current_size - self.pool_in_use)
                # Return connection to pool (no close for in-memory demo)
                self._conns.append(conn)

    def close_all(self):
        with self._lock:
            while self._conns:
                c = self._conns.pop()
                try:
                    c.close()
                except Exception:
                    pass
                self.total_closed += 1
            self.current_size = self.pool_in_use  # keep those checked out as size
            self.pool_available = max(0, self.current_size - self.pool_in_use)

    def stats(self) -> Dict[str, int]:
        return {
            "current_size": int(self.current_size),
            "pool_available": int(self.pool_available),
            "pool_in_use": int(self.pool_in_use),
            "peak_size": int(self.peak_size),
            "total_created": int(self.total_created),
            "total_closed": int(self.total_closed),
            "wait_count": int(self.wait_count),
            "timeout_count": int(self.timeout_count),
        }


class _PoolManager:
    def __init__(self):
        self._db_pools: Dict[str, _SQLitePool] = {}
        self._lock = threading.Lock()

    def create_database_pool(self, name: str, cfg: DatabaseConfig):
        if cfg.db_type != DatabaseType.SQLITE:
            raise ValueError("Only SQLITE is supported in tests")
        with self._lock:
            self._db_pools[name] = _SQLitePool(cfg)

    @contextmanager
    def get_database_connection(self, name: str) -> Iterator[sqlite3.Connection]:
        pool = self._db_pools.get(name)
        if not pool:
            raise KeyError(f"Pool not found: {name}")
        with pool.acquire() as conn:
            yield conn

    def remove_database_pool(self, name: str):
        with self._lock:
            pool = self._db_pools.pop(name, None)
            if pool:
                pool.close_all()

    def get_all_stats(self) -> Dict:
        with self._lock:
            return {
                "database_pools": {k: v.stats() for k, v in self._db_pools.items()},
                "http_pool": None,
            }


# singleton
pool_manager = _PoolManager()
