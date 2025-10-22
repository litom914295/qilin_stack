# -*- coding: utf-8 -*-
"""
预测涨停板.py（新手友好版）

用途：快速跑通“明日涨停板预测”的最小可用示例，便于确认环境没问题。
说明：本示例默认使用“模拟数据 + 随机概率”演示完整流程，确保开箱即跑。
      当你准备好真实数据/模型时，按注释中的“正式使用”步骤替换即可。

使用前准备（一次性）：
1) 建议已在项目根目录创建并激活虚拟环境（见 README 开头“环境准备”）
2) 安装依赖：
   pip install -U pip
   pip install pandas numpy
   # 如需接入真实行情/训练模型，再按 README 安装 akshare/pyqlib/lightgbm/xgboost 等
3) （可选）下载 Qlib 日线数据：
   python scripts/validate_qlib_data.py --download

运行：
   python 预测涨停板.py

期望输出：按概率从高到低列出候选股票，给出关注建议。
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ===============
# Step 0. 项目路径（确保可从项目根目录运行）
# ===============
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===============
# Step 1. 配置与参数（可按需修改）
# ===============
SEED = 42                 # 随机种子（保证可复现的示例结果）
TOPK = 10                 # 展示前多少只股票
STRONG_FOCUS_TH = 0.80    # 强烈关注阈值
FOCUS_TH = 0.70           # 关注阈值

# ===============
# Step 2. 准备股票列表
# ===============
# 你可以把下面的示例股票替换为自己的自选股；格式：6位代码 + .SZ/.SH
stock_list = [
    "000001.SZ", "000002.SZ", "000333.SZ", "000651.SZ", "000768.SZ",
    "002415.SZ", "300750.SZ", "600036.SH", "600519.SH", "601318.SH",
]

# ===============
# Step 3. 准备特征矩阵（示例用随机特征；正式使用请替换为真实特征）
# ===============
np.random.seed(SEED)
# 假设有 50 个特征，真实场景请用你计算好的因子/特征替换
n_features = 50
X_demo = pd.DataFrame(
    np.random.randn(len(stock_list), n_features),
    index=stock_list,
    columns=[f"feature_{i}" for i in range(n_features)],
)

# 正式使用（示例）——用你自己的特征替换：
# 1) 若已计算出特征 CSV：
#    X_demo = pd.read_csv("your_features.csv", index_col=0)
# 2) 若要基于 Qlib 取数并计算特征：
#    - 参考：examples/limitup_example.py 与 factors/limitup_advanced_factors.py
#    - 思路：用 Qlib 读出行情 -> 计算涨停相关因子 -> 拼成特征矩阵 X_demo

# ===============
# Step 4. 载入/构建模型（示例用“模拟概率”，确保可直接运行）
# ===============
# 快速体验：用随机概率模拟一个“模型输出”
proba_demo = np.random.uniform(0.60, 0.95, size=len(stock_list))

# 正式使用 1：载入你训练好的模型（例如 sklearn / LightGBM / XGBoost）
# from joblib import load
# model = load("models/best_model.joblib")
# proba_real = model.predict_proba(X_demo)[:, 1]
#
# 正式使用 2：调用本项目的集成流程（示例）
# - 参考 examples/limitup_example.py（因子发现/模型优化/端到端流程）
# - 也可接入 rd_agent/* 的因子库与一进二研究管线

# ===============
# Step 5. 生成“是否可能涨停”的判断与建议
# ===============
results = (
    pd.DataFrame({
        "stock": stock_list,
        "limit_up_prob": proba_demo,   # 把 proba_real 替换到这里即可
    })
    .sort_values("limit_up_prob", ascending=False)
    .head(TOPK)
    .reset_index(drop=True)
)

# 打标签 + 建议
labels = []
advices = []
for p in results["limit_up_prob"].values:
    if p >= STRONG_FOCUS_TH:
        labels.append("✅ 可能涨停")
        advices.append("强烈关注！")
    elif p >= FOCUS_TH:
        labels.append("✅ 有一定机会")
        advices.append("可以关注")
    else:
        labels.append("❌ 机会较小")
        advices.append("观望")

results["label"] = labels
results["advice"] = advices

# ===============
# Step 6. 打印结果
# ===============
print("\n" + "=" * 60)
print("📊 明日涨停板预测（演示版）")
print("=" * 60)
for i, row in results.iterrows():
    stock = row["stock"]
    prob = row["limit_up_prob"]
    label = row["label"]
    advice = row["advice"]
    print(f"\n{i+1:02d}. {stock}")
    print(f"预测结果：{label}")
    print(f"涨停概率：{prob:.1%}")
    print(f"建议：{advice}")

print("\n" + "=" * 60)
print("✅ 预测完成（演示数据）。")
print("=" * 60)

# ===============
# Step 7. 下一步（如何从演示走向实盘/回测）
# ===============
print("\n下一步建议：")
print("1) 用 Qlib/AKShare 获取真实行情，计算涨停相关因子，替换本示例的随机特征。")
print("2) 训练并保存模型（LightGBM/XGBoost/CatBoost），将 predict_proba 输出接到本脚本。")
print("3) 参考 examples/limitup_example.py 运行端到端流程，逐步替换演示逻辑。")
