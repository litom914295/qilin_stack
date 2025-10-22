#!/bin/bash
# 测试运行脚本

set -e

echo "========================================"
echo "麒麟量化系统测试套件"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python环境
echo -e "${YELLOW}检查Python环境...${NC}"
python --version
pip list | grep pytest || pip install pytest pytest-asyncio pytest-cov pytest-xdist

# 创建日志目录
mkdir -p logs
mkdir -p htmlcov

# 运行测试
echo -e "\n${YELLOW}========== 1. 单元测试 ==========${NC}"
pytest tests/unit/ -v --tb=short -m "not slow" || true

echo -e "\n${YELLOW}========== 2. 集成测试 ==========${NC}"
pytest tests/integration/ -v --tb=short -m integration || true

echo -e "\n${YELLOW}========== 3. MLOps测试 ==========${NC}"
pytest tests/unit/test_mlops.py -v --tb=short -m mlops || true

echo -e "\n${YELLOW}========== 4. 监控测试 ==========${NC}"
pytest tests/unit/test_monitoring.py -v --tb=short -m monitoring || true

echo -e "\n${YELLOW}========== 5. 覆盖率报告 ==========${NC}"
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing --cov-report=xml

# 检查覆盖率
coverage_pct=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')

echo -e "\n${YELLOW}========== 测试总结 ==========${NC}"
echo -e "代码覆盖率: ${coverage_pct}%"

if (( $(echo "$coverage_pct >= 80" | bc -l) )); then
    echo -e "${GREEN}✓ 覆盖率达标 (>= 80%)${NC}"
    exit 0
else
    echo -e "${RED}✗ 覆盖率不达标 (< 80%)${NC}"
    echo -e "${YELLOW}请查看 htmlcov/index.html 了解详情${NC}"
    exit 1
fi
