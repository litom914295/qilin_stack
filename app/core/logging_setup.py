"""
轻量级日志配置工具
"""
from __future__ import annotations
import logging
from typing import Optional

def setup_logging(level: str = "INFO", fmt: Optional[str] = None):
    """Configure root logging once.
    Safe to call multiple times; only configures when no handlers present.
    """
    if logging.getLogger().handlers:
        return
    fmt = fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)