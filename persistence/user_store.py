"""
用户账户与使用日志的简单持久化（本地JSON + 内存缓存）
- 支持多用户点数、等级管理
- 支持使用日志追加与查询
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_STORE_PATH = Path("persistence/user_store.json")


def _now_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


@dataclass
class User:
    user_id: str
    points: int = 0
    level: str = "VIP"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UsageLog:
    date: str
    op: str
    stocks: int
    points: int
    user_id: str


@dataclass
class UserStore:
    users: Dict[str, User] = field(default_factory=dict)
    logs: List[UsageLog] = field(default_factory=list)

    def ensure_user(self, user_id: str, default_points: int = 0, level: str = "VIP") -> User:
        if user_id not in self.users:
            self.users[user_id] = User(user_id=user_id, points=int(default_points), level=level)
            self._persist()
        return self.users[user_id]

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def add_points(self, user_id: str, delta: int) -> int:
        u = self.ensure_user(user_id)
        u.points = max(0, int(u.points) + int(delta))
        self._persist()
        return u.points

    def set_points(self, user_id: str, points: int) -> int:
        u = self.ensure_user(user_id)
        u.points = max(0, int(points))
        self._persist()
        return u.points

    def append_log(self, user_id: str, op: str, stocks: int, points: int, date: Optional[str] = None) -> None:
        entry = UsageLog(date=date or _now_date(), op=op, stocks=int(stocks), points=int(points), user_id=user_id)
        self.logs.append(entry)
        self._persist()

    def get_logs(self, user_id: Optional[str] = None, limit: int = 100) -> List[UsageLog]:
        items = self.logs
        if user_id:
            items = [x for x in items if x.user_id == user_id]
        return list(reversed(items[-limit:]))

    def total_points_used(self, user_id: Optional[str] = None) -> int:
        items = self.logs
        if user_id:
            items = [x for x in items if x.user_id == user_id]
        return sum(int(x.points) for x in items)

    def _persist(self) -> None:
        try:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _STORE_PATH.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "users": {uid: asdict(u) for uid, u in self.users.items()},
                        "logs": [asdict(l) for l in self.logs],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    @classmethod
    def load(cls) -> "UserStore":
        if _STORE_PATH.exists():
            try:
                with _STORE_PATH.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                users = {uid: User(**ud) for uid, ud in obj.get("users", {}).items()}
                logs = [UsageLog(**ld) for ld in obj.get("logs", [])]
                return cls(users=users, logs=logs)
            except Exception:
                return cls()
        return cls()


_store_singleton: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = UserStore.load()
    return _store_singleton
