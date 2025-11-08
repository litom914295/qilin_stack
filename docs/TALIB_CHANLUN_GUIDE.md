# TA-Lib 集成与缠论形态使用指南

## 📋 目录

1. [功能概述](#功能概述)
2. [核心组件](#核心组件)
3. [快速开始](#快速开始)
4. [缠论形态识别](#缠论形态识别)
5. [UI界面使用](#ui界面使用)
6. [进阶用法](#进阶用法)

---

## 功能概述

麒麟量化系统现已完整集成TA-Lib技术指标库和缠论形态识别功能：

### ✅ 已实现功能

| 功能模块 | 状态 | 说明 |
|---------|------|------|
| **TA-Lib技术指标** | ✅ | 150+技术指标（趋势、动量、波动率、成交量） |
| **K线形态识别** | ✅ | 100+K线形态（十字星、锤子线、吞没等） |
| **缠论形态** | ✅ | 笔、段级别形态识别 |
| **Qlib集成** | ✅ | 无缝集成到Qlib工作流 |
| **一进二优化** | ✅ | 涨停板专用Handler |

### 🎯 适用场景

- **一进二涨停板策略** - 使用缠论形态识别买点
- **技术分析建模** - 基于TA-Lib指标训练模型
- **形态学回测** - K线形态统计与验证
- **混合特征工程** - Qlib + TA-Lib混合特征

---

## 核心组件

### 1. TA-Lib指标包装器

**文件**: `features/talib_indicators.py`

```python
from features.talib_indicators import TALibIndicators, TALibPatterns, TALibFeatureGenerator

# 计算单个指标
indicators = TALibIndicators()
rsi = indicators.RSI(close_prices, timeperiod=14)
macd, signal, hist = indicators.MACD(close_prices)

# K线形态识别
patterns = TALibPatterns()
doji = patterns.CDLDOJI(open_, high, low, close)
hammer = patterns.CDLHAMMER(open_, high, low, close)

# 缠论形态
bi_patterns = patterns.detect_bi_pattern(open_, high, low, close)
# 返回: {'top_reversal', 'bottom_reversal', 'continuation_up', 'continuation_down'}

# 一键生成所有特征
generator = TALibFeatureGenerator(include_patterns=True)
features_df = generator.generate_features(df)  # 输入OHLCV → 输出49个特征
```

### 2. Qlib Handler集成

**文件**: `qlib_enhanced/talib_handler.py`

```python
# 三种Handler可用:

# 1. 纯TA-Lib特征
from qlib_enhanced.talib_handler import TALibHandler

# 2. Alpha360 + TA-Lib混合
from qlib_enhanced.talib_handler import TALibAlpha360Handler

# 3. 涨停板专用（含缠论形态）
from qlib_enhanced.talib_handler import LimitUpTALibHandler
```

### 3. 模板配置

**文件**: `configs/qlib_workflows/templates/limitup_talib_chanlun.yaml`

- 使用 `LimitUpTALibHandler`
- 包含趋势、动量、波动率、成交量、K线形态5大类特征
- 缠论笔、段形态自动识别
- 针对一进二场景优化的标签

---

## 快速开始

### 方式1: 使用Qlib模板（推荐）

```yaml
# configs/qlib_workflows/templates/my_limitup.yaml

task:
    dataset:
        handler:
            class: LimitUpTALibHandler
            module_path: qlib_enhanced.talib_handler
            kwargs:
                start_time: "2015-01-01"
                end_time: "2023-12-31"
                instruments: "csi300"
                include_patterns: true  # 包含K线形态
                feature_groups:
                    - trend      # SMA, EMA, MACD, ADX
                    - momentum   # RSI, STOCH, CCI, MOM
                    - volatility # ATR, BBANDS, NATR
                    - volume     # OBV, AD, MFI
                    - patterns   # K线形态 + 缠论形态
```

### 方式2: Python代码直接使用

```python
import pandas as pd
from features.talib_indicators import TALibFeatureGenerator

# 准备OHLCV数据
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 生成特征
generator = TALibFeatureGenerator(include_patterns=True)
features = generator.generate_features(df)

print(f"生成 {len(features.columns)} 个特征")
print(features.head())
```

### 方式3: Web界面训练

1. 打开麒麟量化Web界面
2. 导航到"模型库" → "Qlib Workflow"
3. 选择模板: **"一进二涨停（TA-Lib+缠论）"**
4. 点击"训练模型"

---

## 缠论形态识别

### 📊 缠论笔形态（Bi Pattern）

识别笔级别的转折点和延续形态：

```python
from features.talib_indicators import TALibPatterns

patterns = TALibPatterns()
bi_patterns = patterns.detect_bi_pattern(open_, high, low, close)

# 返回字典:
{
    'top_reversal': array([...]),      # 顶部反转信号（100=强, -100=弱）
    'bottom_reversal': array([...]),   # 底部反转信号
    'continuation_up': array([...]),   # 上涨延续信号
    'continuation_down': array([...])  # 下跌延续信号
}
```

**组成形态**:
- **顶部反转**: 射击之星 + 黄昏之星 + 吊颈线
- **底部反转**: 锤子线 + 早晨之星 + 倒锤子线
- **延续形态**: 三白兵（上涨）/ 三只乌鸦（下跌）

### 📈 缠论段形态（Duan Pattern）

识别段级别的结构：

```python
duan_patterns = patterns.detect_duan_pattern(open_, high, low, close)

# 返回字典:
{
    'strong_reversal': array([...]),  # 强反转（吞没形态）
    'weak_reversal': array([...])     # 弱反转（孕线、十字星）
}
```

### 🎯 一进二应用示例

```python
import pandas as pd
import numpy as np

# 加载涨停板股票数据
df = load_limitup_stocks()

# 识别缠论形态
bi_patterns = TALibPatterns.detect_bi_pattern(
    df['open'].values,
    df['high'].values,
    df['low'].values,
    df['close'].values
)

# 策略逻辑: 底部反转 + RSI超卖 = 买入信号
rsi = TALibIndicators.RSI(df['close'].values, 14)
buy_signal = (bi_patterns['bottom_reversal'] > 0) & (rsi < 30)

# 找出符合条件的股票
df['buy_signal'] = buy_signal
candidates = df[df['buy_signal'] == True]
print(f"找到 {len(candidates)} 个买点")
```

---

## UI界面使用

### 步骤1: 启动Web界面

```bash
# 激活虚拟环境
.\.qilin\Scripts\activate

# 启动Streamlit
streamlit run web/unified_dashboard.py
```

### 步骤2: 选择模板

导航路径: **"模型库"** → **"Qlib Workflow"** → **"模板管理"**

找到模板: `limitup_talib_chanlun.yaml`

**模板特点**:
- ✅ 使用 `LimitUpTALibHandler`
- ✅ 包含49个TA-Lib特征
- ✅ 缠论笔、段形态自动识别
- ✅ 针对涨停板优化的标签

### 步骤3: 训练模型

1. 点击"加载模板"
2. 确认配置参数
3. 点击"开始训练"
4. 等待训练完成
5. 查看回测结果

### 步骤4: 查看特征重要性

训练完成后，查看哪些TA-Lib特征对一进二预测最有效：

```
Top 10 重要特征:
1. rsi_6              - RSI(6)相对强弱指标
2. bi_bottom_reversal - 缠论笔底部反转
3. macd_hist          - MACD柱状图
4. atr_14             - ATR真实波幅
5. bbands_width       - 布林带宽度
6. hammer             - 锤子线形态
7. volume_ratio       - 量比
8. stoch_k            - 随机指标K值
9. duan_strong_reversal - 缠论段强反转
10. mfi_14            - 资金流量指标
```

---

## 进阶用法

### 自定义特征组

只使用部分TA-Lib特征：

```yaml
handler:
    class: TALibHandler
    module_path: qlib_enhanced.talib_handler
    kwargs:
        include_patterns: true
        feature_groups:
            - momentum   # 只用动量指标
            - patterns   # 和K线形态
```

### 混合Alpha360特征

结合Qlib内置特征和TA-Lib特征：

```yaml
handler:
    class: TALibAlpha360Handler
    module_path: qlib_enhanced.talib_handler
    kwargs:
        include_patterns: false  # Alpha360已有足够特征
```

### 自定义缠论形态

扩展缠论形态识别：

```python
from features.talib_indicators import TALibPatterns
import talib

class MyChanlunPatterns(TALibPatterns):
    @staticmethod
    def detect_zhongshu(open_, high, low, close):
        """识别缠论中枢"""
        # 自定义中枢识别逻辑
        # ...
        return zhongshu_signal
```

### 单独计算某个指标

```python
from features.talib_indicators import calculate_indicator, detect_pattern

# 计算RSI
rsi = calculate_indicator(df, 'RSI', timeperiod=14)

# 检测十字星
doji = detect_pattern(df, 'CDLDOJI')
```

---

## 📚 可用特征列表

### 趋势指标（11个）

| 特征名 | 说明 | 参数 |
|-------|------|------|
| sma_5, sma_10, sma_20, sma_60 | 简单移动平均 | 周期5/10/20/60 |
| ema_5, ema_10, ema_20 | 指数移动平均 | 周期5/10/20 |
| macd, macd_signal, macd_hist | MACD指标 | 12/26/9 |
| adx_14 | 平均趋向指数 | 周期14 |

### 动量指标（9个）

| 特征名 | 说明 | 参数 |
|-------|------|------|
| rsi_6, rsi_14, rsi_24 | 相对强弱指标 | 周期6/14/24 |
| stoch_k, stoch_d | 随机指标 | KD值 |
| cci_14 | 顺势指标 | 周期14 |
| mom_10 | 动量指标 | 周期10 |
| roc_10 | 变动率指标 | 周期10 |
| willr_14 | 威廉指标 | 周期14 |

### 波动率指标（6个）

| 特征名 | 说明 | 参数 |
|-------|------|------|
| atr_14, natr_14 | 真实波幅 | 周期14 |
| bbands_upper, bbands_middle, bbands_lower | 布林带 | 20/2.0 |
| bbands_width | 布林带宽度 | - |

### 成交量指标（3个）

| 特征名 | 说明 |
|-------|------|
| obv | 能量潮 |
| ad | 累积/派发线 |
| mfi_14 | 资金流量指标 |

### K线形态（11个）

| 特征名 | 说明 | 信号 |
|-------|------|------|
| doji | 十字星 | 反转 |
| hammer | 锤子线 | 底部反转 |
| inverted_hammer | 倒锤子线 | 底部反转 |
| hanging_man | 吊颈线 | 顶部反转 |
| shooting_star | 射击之星 | 顶部反转 |
| engulfing | 吞没形态 | 反转 |
| harami | 孕线形态 | 反转 |
| morning_star | 早晨之星 | 底部反转 |
| evening_star | 黄昏之星 | 顶部反转 |
| three_white_soldiers | 三白兵 | 上涨延续 |
| three_black_crows | 三只乌鸦 | 下跌延续 |

### 缠论形态（6个）

| 特征名 | 说明 | 级别 |
|-------|------|------|
| bi_top_reversal | 笔顶部反转 | 笔 |
| bi_bottom_reversal | 笔底部反转 | 笔 |
| bi_continuation_up | 笔上涨延续 | 笔 |
| bi_continuation_down | 笔下跌延续 | 笔 |
| duan_strong_reversal | 段强反转 | 段 |
| duan_weak_reversal | 段弱反转 | 段 |

**总计: 49个特征**

---

## 🔧 故障排查

### 问题1: 导入TA-Lib失败

```python
ImportError: DLL load failed while importing _ta_lib
```

**解决方案**:
```bash
# 重新安装TA-Lib
pip install TA-Lib==0.4.32
```

### 问题2: Qlib Handler找不到

```
ModuleNotFoundError: No module named 'qlib_enhanced.talib_handler'
```

**解决方案**:
确保项目根目录在Python路径中：
```python
import sys
sys.path.append('G:/test/qilin_stack')
```

### 问题3: 特征计算NaN过多

**原因**: TA-Lib指标需要预热期（如MA(60)需要至少60根K线）

**解决方案**:
```python
# 在Handler中添加Fillna处理器
infer_processors = [
    {"class": "Fillna", "kwargs": {"fields_group": "feature", "fill_value": 0}},
]
```

---

## 📞 技术支持

- **文档**: `docs/TALIB_CHANLUN_GUIDE.md`
- **示例代码**: `features/talib_indicators.py`
- **模板配置**: `configs/qlib_workflows/templates/limitup_talib_chanlun.yaml`

---

## 🎉 总结

TA-Lib已完整集成到麒麟量化系统：

✅ **150+技术指标** - 覆盖趋势、动量、波动率、成交量  
✅ **100+K线形态** - 自动识别经典形态  
✅ **缠论形态** - 笔、段级别结构识别  
✅ **Qlib无缝集成** - 直接用于模型训练  
✅ **一进二优化** - 涨停板专用Handler  

现在你可以在一进二涨停选股中使用缠论形态学了！🚀
