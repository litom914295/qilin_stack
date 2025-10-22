"""
生成 SLO 验收报告（JSON）
- 复用测试中的 E2ESLOValidator，运行端到端/故障恢复/并发负载三类用例
- 输出 JSON 报告到 reports/slo_report_YYYYMMDD_HHMMSS.json
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

# 设定工作目录与路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
E2E_TEST = TESTS_DIR / "e2e" / "test_mvp_slo.py"

# 确保可以 import 测试内的 E2ESLOValidator
sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.test_mvp_slo import E2ESLOValidator  # type: ignore


def main() -> int:
    validator = E2ESLOValidator()

    # 1. 端到端流程（20只股票）
    test_stocks = [f"{i:06d}" for i in range(1, 21)]
    import asyncio

    asyncio.run(validator.test_end_to_end_flow(test_stocks))
    # 2. 故障恢复
    asyncio.run(validator.test_failover_recovery())
    # 3. 并发负载
    asyncio.run(validator.test_concurrent_load(concurrent_stocks=100))

    report = validator.generate_slo_report()

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"slo_report_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"SLO report written: {out_path}")
    return 0 if report["summary"]["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
