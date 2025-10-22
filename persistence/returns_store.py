"""
简单的收益回放存储（本地JSON + 内存缓存）
供自适应权重服务评估使用：记录每只股票的已实现收益（如T+1或平仓收益）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_STORE_PATH = Path("persistence/returns_store.json")


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class ReturnsStore:
    data: Dict[str, Dict[str, float]] = field(default_factory=dict)  # date -> {symbol: return}

    def record(self, symbol: str, realized_return: float, date: Optional[str] = None) -> None:
        d = date or datetime.now().strftime("%Y-%m-%d")
        self.data.setdefault(d, {})[symbol] = float(realized_return)
        self._persist()

    def record_batch(self, returns: Dict[str, float], date: Optional[str] = None) -> None:
        d = date or datetime.now().strftime("%Y-%m-%d")
        self.data.setdefault(d, {}).update({k: float(v) for k, v in returns.items()})
        self._persist()

    def get_by_date(self, date: str) -> Dict[str, float]:
        return dict(self.data.get(date, {}))

    def get_recent(self, days: int = 3) -> Dict[str, float]:
        # 合并最近N天（同一symbol取最近一次）
        out: Dict[str, float] = {}
        for i in range(days):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            for sym, r in self.data.get(d, {}).items():
                if sym not in out:
                    out[sym] = float(r)
        return out

    def _persist(self) -> None:
        try:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _STORE_PATH.open("w", encoding="utf-8") as f:
                json.dump({k: v for k, v in self.data.items()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @classmethod
    def load(cls) -> "ReturnsStore":
        if _STORE_PATH.exists():
            try:
                with _STORE_PATH.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                return cls(data={k: {sk: float(sv) for sk, sv in v.items()} for k, v in obj.items()})
            except Exception:
                return cls()
        return cls()


_store_singleton: Optional[ReturnsStore] = None


def get_returns_store() -> ReturnsStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = ReturnsStore.load()
    return _store_singleton
