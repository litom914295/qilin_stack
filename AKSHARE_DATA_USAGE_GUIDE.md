# 麒麟量化系统 - AKShare真实数据使用指南

## 📦 环境准备

### 安装依赖

```bash
pip install akshare>=1.10.0
pip install lightgbm>=4.0.0
pip install optuna>=3.0.0
pip install scikit-learn>=1.0.0
pip install tqdm>=4.60.0
```

---

## 🔄 完整数据流程

### 1️⃣ 历史数据收集与标注

使用AKShare收集历史涨停数据并自动标注首板→二板成功率。

#### 基础用法

```python
from app.data_collector import HistoricalDataCollector
import logging

logging.basicConfig(level=logging.INFO)

# 初始化收集器(use_akshare=True使用真实数据)
collector = HistoricalDataCollector(
    output_dir="data/historical",
    use_akshare=True  # 使用AKShare真实数据
)

# 收集某日的涨停数据
date = "2024-01-15"
limitup_stocks = collector.collect_daily_limitup_stocks(date)

print(f"收集到 {len(limitup_stocks)} 只涨停股票")
for stock in limitup_stocks[:3]:
    print(f"  {stock['symbol']} {stock['name']}: "
          f"封板时间={stock['limit_up_time']}, "
          f"打开次数={stock['open_times']}")

# 标注首板→二板成功率
symbol = limitup_stocks[0]['symbol']
label = collector.label_first_to_second_board(
    first_board_date=date,
    symbol=symbol,
    stock_data=limitup_stocks[0]
)

print(f"\n{symbol} 次日涨停标签: {label} ({'成功' if label == 1 else '失败'})")
```

#### 批量收集数据集

```python
# 收集近3个月的历史涨停数据
df = collector.collect_and_label_dataset(
    start_date='2024-01-01',
    end_date='2024-03-31',
    save_path='data/historical/training_data_Q1_2024.csv'
)

print(f"\n数据集总样本数: {len(df)}")
print(f"成功率: {df['label'].mean():.2%}")
print(f"\n特征维度: {len(df.columns) - 4}")  # 减去date, symbol, name, label
print(f"特征列: {list(df.columns)}")
```

---

### 2️⃣ LightGBM模型训练

使用收集的真实数据训练LightGBM模型。

#### 训练流程

```python
from app.lgb_trainer import LightGBMTrainer
from app.data_collector import HistoricalDataCollector

# Step 1: 收集数据
collector = HistoricalDataCollector(use_akshare=True)

df = collector.collect_and_label_dataset(
    start_date='2023-01-01',  # 建议至少1年数据
    end_date='2023-12-31'
)

print(f"数据集样本数: {len(df)}")
print(f"正负样本比例: {df['label'].value_counts()}")

# Step 2: 训练模型
trainer = LightGBMTrainer(model_dir="models")

# 准备数据(按时间划分)
X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
    df,
    test_size=0.2,
    time_split=True  # ⚠️ 重要: 按时间划分避免未来数据泄漏
)

# 训练
model = trainer.train(
    X_train, y_train,
    X_test, y_test,
    feature_names=feature_names
)

# Step 3: 评估
metrics = trainer.evaluate(X_test, y_test)

print(f"\n模型性能:")
print(f"  AUC: {metrics['auc']:.4f}")
print(f"  准确率: {metrics['accuracy']:.2%}")
print(f"  召回率: {metrics['recall']:.2%}")

# Step 4: 保存模型
model_path = trainer.save_model()
print(f"\n模型已保存: {model_path}")
```

#### 超参数优化(可选)

```python
# 使用Optuna自动调参(⚠️ 耗时较长,建议50-100轮)
best_params = trainer.optimize_hyperparameters(
    X_train, y_train,
    feature_names=feature_names,
    n_trials=50
)

# 使用最佳参数重新训练
model = trainer.train(
    X_train, y_train,
    X_test, y_test,
    params=best_params,
    feature_names=feature_names
)

metrics = trainer.evaluate(X_test, y_test)
print(f"\n优化后AUC: {metrics['auc']:.4f}")
```

---

### 3️⃣ 回测系统验证

使用历史信号数据进行回测,计算Sharpe比率、最大回撤等指标。

#### 生成回测信号

```python
from app.lgb_trainer import LightGBMTrainer
from app.data_collector import HistoricalDataCollector
import pandas as pd

# 加载训练好的模型
trainer = LightGBMTrainer()
trainer.load_model("models/lgb_model_20240129_120000.txt")

# 收集回测期数据
collector = HistoricalDataCollector(use_akshare=True)

backtest_data = []
dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')

for date in dates:
    date_str = date.strftime('%Y-%m-%d')
    
    # 获取当日涨停股票
    limitup_stocks = collector.collect_daily_limitup_stocks(date_str)
    
    if not limitup_stocks:
        continue
    
    for stock in limitup_stocks:
        # 提取特征
        features = collector.extract_features_from_dict(stock)
        
        # 预测概率
        X = [list(features.values())]
        prob = trainer.predict(X)[0]
        
        backtest_data.append({
            'date': date_str,
            'symbol': stock['symbol'],
            'score': prob * 100,  # 转换为0-100分数
            'price': stock['price']
        })

signals_df = pd.DataFrame(backtest_data)
signals_df.to_csv('data/backtest_signals.csv', index=False)

print(f"生成 {len(signals_df)} 条回测信号")
```

#### 运行回测

```python
from app.backtest_engine import BacktestEngine, BacktestConfig

# 配置回测参数
config = BacktestConfig(
    start_date='2024-01-01',
    end_date='2024-06-30',
    initial_capital=100000,      # 10万本金
    top_k=5,                     # 每日买入Top5
    commission_rate=0.0003,      # 万三佣金
    slippage=0.01,               # 1%滑点
    stop_loss=-0.03,             # -3%止损
    take_profit=0.10             # 10%止盈
)

# 运行回测
engine = BacktestEngine(config)
metrics = engine.run_backtest(signals_df)

# 查看结果
print("\n" + "=" * 60)
print("回测结果")
print("=" * 60)
print(f"总收益率: {metrics['total_return']:.2%}")
print(f"年化收益率: {metrics['annualized_return']:.2%}")
print(f"Sharpe比率: {metrics['sharpe_ratio']:.4f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
print(f"胜率: {metrics['win_rate']:.2%}")
print(f"总交易次数: {metrics['total_trades']}")
print("=" * 60)

# 保存详细报告
engine.save_results("reports/backtest")
```

---

## 🔧 实战示例

### 完整训练+回测流程

```python
#!/usr/bin/env python
"""
完整的数据收集→模型训练→回测验证流程
"""
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def main():
    print("=" * 60)
    print("麒麟量化系统 - 完整训练与回测流程")
    print("=" * 60)
    
    # ========== Step 1: 数据收集 ==========
    print("\n[1/4] 收集历史数据...")
    
    from app.data_collector import HistoricalDataCollector
    
    collector = HistoricalDataCollector(use_akshare=True)
    
    # 收集2023年全年数据作为训练集
    df_train = collector.collect_and_label_dataset(
        start_date='2023-01-01',
        end_date='2023-12-31',
        save_path='data/historical/train_2023.csv'
    )
    
    print(f"✓ 训练数据: {len(df_train)} 样本")
    print(f"  正样本(次日涨停): {df_train['label'].sum()}")
    print(f"  负样本(次日未涨停): {(1-df_train['label']).sum()}")
    print(f"  成功率: {df_train['label'].mean():.2%}")
    
    # ========== Step 2: 模型训练 ==========
    print("\n[2/4] 训练LightGBM模型...")
    
    from app.lgb_trainer import LightGBMTrainer
    
    trainer = LightGBMTrainer()
    
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(
        df_train,
        time_split=True
    )
    
    model = trainer.train(
        X_train, y_train,
        X_test, y_test,
        feature_names=feature_names
    )
    
    metrics = trainer.evaluate(X_test, y_test)
    
    print(f"✓ 模型训练完成")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  准确率: {metrics['accuracy']:.2%}")
    
    model_path = trainer.save_model()
    
    # ========== Step 3: 生成回测信号 ==========
    print("\n[3/4] 生成回测信号...")
    
    import pandas as pd
    
    signals = []
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        limitup_stocks = collector.collect_daily_limitup_stocks(date_str)
        
        for stock in limitup_stocks:
            features = collector.extract_features_from_dict(stock)
            X = [list(features.values())]
            prob = trainer.predict(X)[0]
            
            signals.append({
                'date': date_str,
                'symbol': stock['symbol'],
                'score': prob * 100,
                'price': stock['price']
            })
    
    signals_df = pd.DataFrame(signals)
    
    print(f"✓ 生成 {len(signals_df)} 条信号")
    
    # ========== Step 4: 回测 ==========
    print("\n[4/4] 运行回测...")
    
    from app.backtest_engine import BacktestEngine, BacktestConfig
    
    config = BacktestConfig(
        start_date='2024-01-01',
        end_date='2024-06-30',
        initial_capital=100000,
        top_k=5
    )
    
    engine = BacktestEngine(config)
    results = engine.run_backtest(signals_df)
    
    print(f"\n✓ 回测完成")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  Sharpe比率: {results['sharpe_ratio']:.4f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    
    engine.save_results("reports/backtest")
    
    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

保存为 `scripts/train_and_backtest.py` 并运行:

```bash
python scripts/train_and_backtest.py
```

---

## ⚠️ 注意事项

### AKShare数据限制

1. **频率限制**
   - AKShare免费接口有请求频率限制
   - 建议收集数据时加延时: `time.sleep(0.5)`

2. **历史数据范围**
   - `stock_zt_pool_em()` 仅支持最近2年数据
   - 更早数据需要使用其他接口或数据源

3. **数据质量**
   - 部分字段可能缺失(如`首次封板时间`)
   - 需要做缺失值处理

### 训练建议

1. **样本量**
   - 建议至少1000个样本
   - 最好1年以上历史数据

2. **正负样本比例**
   - 首板→二板成功率通常20-40%
   - 可使用SMOTE等方法平衡样本

3. **特征工程**
   - 16维特征已经比较完善
   - 可根据需要增加板块轮动、市场情绪等特征

### 回测建议

1. **真实成本**
   - 考虑佣金、滑点、涨停板买不到的概率
   - 建议滑点设置1-2%

2. **避免未来函数**
   - 训练集/测试集按时间划分
   - 不使用未来数据

3. **稳定性验证**
   - 在多个时间段回测
   - 关注最大回撤和夏普比率

---

## 📊 数据字段说明

### AKShare涨停池字段映射

| AKShare字段 | 系统字段 | 说明 |
|------------|---------|------|
| 代码 | symbol | 股票代码 |
| 名称 | name | 股票名称 |
| 最新价 | price | 当前价格 |
| 涨跌幅 | change_pct | 涨幅(%) |
| 换手率 | turnover_rate | 换手率(%) |
| 量比 | volume_ratio | 量比 |
| 首次封板时间 | limit_up_time | 封板时间 |
| 打开次数 | open_times | 打板次数 |
| 封板资金 | seal_amount | 封单金额(万) |
| 流通市值 | total_amount | 流通市值(亿) |
| 所属行业 | sector | 行业/板块 |
| 涨停原因 | reason | 题材/概念 |

---

## 🚀 快速测试

### 测试数据收集

```python
from app.data_collector import HistoricalDataCollector

collector = HistoricalDataCollector(use_akshare=True)

# 收集最近一个交易日的涨停数据
import datetime
today = datetime.datetime.now().strftime('%Y-%m-%d')

stocks = collector.collect_daily_limitup_stocks(today)

print(f"今日涨停: {len(stocks)} 只")
for stock in stocks[:5]:
    print(f"  {stock['symbol']} {stock['name']}: {stock['reason']}")
```

### 测试模型训练

```bash
# 使用模拟数据快速测试
python app/lgb_trainer.py
```

### 测试回测引擎

```bash
# 使用模拟信号快速测试
python app/backtest_engine.py
```

---

## 📚 相关文档

- [高优先级任务完成报告](HIGH_PRIORITY_IMPROVEMENTS_COMPLETED.md)
- [中优先级任务完成报告](MEDIUM_PRIORITY_TASKS_COMPLETED.md)
- [AKShare官方文档](https://akshare.akfamily.xyz/)

---

**现在你可以使用AKShare真实数据训练模型了! 🎉**
