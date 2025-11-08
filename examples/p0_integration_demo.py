"""P0增强模块集成示例

演示如何使用新增的P0模块:
- P0-1: TrendClassifier (走势类型识别)
- P0-2: DivergenceDetector (背驰识别)
- P0-3: IntervalTrapStrategy (区间套策略)
- P0-4: ChanLunChartComponent (图表组件)
- P0-6: ChanLunBacktester (回测框架)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging

# 导入P0模块
from qlib_enhanced.chanlun.trend_classifier import TrendClassifier
from qlib_enhanced.chanlun.divergence_detector import DivergenceDetector
from qlib_enhanced.chanlun.interval_trap import IntervalTrapStrategy
from web.components.chanlun_chart import ChanLunChartComponent
from backtest.chanlun_backtest import ChanLunBacktester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_data(days=100):
    """生成模拟数据"""
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # 模拟价格走势
    np.random.seed(42)
    prices = 10 + np.random.randn(days).cumsum() * 0.1
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.randn(days) * 0.01),
        'high': prices * (1 + abs(np.random.randn(days)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(days)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
        # 模拟缠论特征
        'seg_direction': np.random.choice([-1, 1], days),
        'is_buy_point': np.random.choice([0, 1], days, p=[0.95, 0.05]),
        'is_sell_point': np.random.choice([0, 1], days, p=[0.95, 0.05]),
        'bsp_type': np.random.choice([0, 1, 2, 3], days, p=[0.90, 0.03, 0.04, 0.03]),
        '$bi_power': np.abs(np.random.randn(days) * 0.05),
        '$bi_direction': np.random.choice([-1, 1], days),
    })
    
    return df

def demo_p0_1_trend_classifier():
    """示例: P0-1 走势类型识别"""
    print("\n" + "="*60)
    print("P0-1: 走势类型识别")
    print("="*60)
    
    classifier = TrendClassifier()
    
    # 模拟线段和中枢数据
    class MockSeg:
        def __init__(self, direction):
            self._direction = direction
        def is_up(self):
            return self._direction > 0
        def get_begin_val(self):
            return 10 if self._direction > 0 else 12
        def get_end_val(self):
            return 12 if self._direction > 0 else 10
    
    class MockZS:
        def __init__(self):
            self.low = 10.5
            self.high = 11.5
    
    # 测试上涨趋势
    seg_list = [MockSeg(1), MockSeg(1), MockSeg(1)]
    zs_list = [MockZS()]
    
    trend = classifier.classify_trend(seg_list, zs_list)
    print(f"趋势类型: {trend.name}")
    
    result = classifier.classify_with_details(seg_list, zs_list)
    print(f"详细分析: {result}")

def demo_p0_2_divergence_detector():
    """示例: P0-2 背驰识别"""
    print("\n" + "="*60)
    print("P0-2: 背驰识别")
    print("="*60)
    
    detector = DivergenceDetector()
    df = generate_mock_data(100)
    
    # 测试背驰检测
    alpha = detector.calculate_divergence_alpha(df)
    print(f"背驰Alpha均值: {alpha.mean():.4f}")
    print(f"背驰信号数: {(alpha != 0).sum()}")

def demo_p0_3_interval_trap():
    """示例: P0-3 区间套策略"""
    print("\n" + "="*60)
    print("P0-3: 区间套策略")
    print("="*60)
    
    strategy = IntervalTrapStrategy(use_15m=False)
    
    # 构造多级别数据
    multi_level_data = {
        'day': generate_mock_data(100),
        '60m': generate_mock_data(500),
    }
    
    signals = strategy.find_interval_trap_signals(multi_level_data, lookback_days=10)
    print(f"发现区间套信号数: {len(signals)}")
    
    if len(signals) > 0:
        sig = signals[0]
        print(f"示例信号: {sig.reason}, 强度={sig.strength}")

def demo_p0_4_chart():
    """示例: P0-4 图表组件"""
    print("\n" + "="*60)
    print("P0-4: 缠论图表组件")
    print("="*60)
    
    chart = ChanLunChartComponent(width=1400, height=900)
    df = generate_mock_data(60)
    
    # 准备缠论特征
    chan_features = {
        'fx_mark': pd.Series([1 if i % 10 == 0 else -1 if i % 10 == 5 else 0 for i in range(len(df))]),
        'buy_points': [
            {'datetime': df.iloc[10]['datetime'], 'price': df.iloc[10]['close'], 'type': 1},
            {'datetime': df.iloc[30]['datetime'], 'price': df.iloc[30]['close'], 'type': 2},
        ],
        'sell_points': [],
    }
    
    fig = chart.render_chanlun_chart(df, chan_features)
    print(f"图表创建完成: {len(fig.data)} traces")
    
    # 保存为HTML
    output_path = Path(__file__).parent.parent / 'temp' / 'chanlun_chart_demo.html'
    output_path.parent.mkdir(exist_ok=True)
    fig.write_html(str(output_path))
    print(f"图表已保存: {output_path}")

def demo_p0_6_backtest():
    """示例: P0-6 回测框架"""
    print("\n" + "="*60)
    print("P0-6: 回测框架")
    print("="*60)
    
    # 简单策略: 买点买入,10天后卖出
    def simple_strategy(df):
        if len(df) < 2:
            return 'hold'
        
        last_row = df.iloc[-1]
        if last_row.get('is_buy_point', 0) == 1:
            return 'buy'
        
        # 持有10天后卖出
        if len(df) >= 10:
            return 'sell'
        
        return 'hold'
    
    # 创建回测器
    backtester = ChanLunBacktester(initial_cash=1000000, commission_rate=0.0003)
    
    # 生成测试数据
    stock_data = generate_mock_data(100)
    
    # 运行回测
    results = backtester.backtest_strategy(
        strategy=simple_strategy,
        stock_data=stock_data,
        start_date='2023-01-01',
        end_date='2023-04-10'
    )
    
    # 显示结果
    if results:
        metrics = results['metrics']
        print(f"总收益率: {metrics.total_return:.2%}")
        print(f"年化收益: {metrics.annual_return:.2%}")
        print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
        print(f"最大回撤: {metrics.max_drawdown:.2%}")
        print(f"胜率: {metrics.win_rate:.2%}")
        print(f"盈亏比: {metrics.profit_factor:.2f}")
        print(f"总交易次数: {metrics.total_trades}")

def demo_integration():
    """完整集成示例"""
    print("\n" + "="*60)
    print("完整集成示例")
    print("="*60)
    
    # 1. 生成数据
    df = generate_mock_data(100)
    print(f"✓ 生成数据: {len(df)}条")
    
    # 2. 走势识别
    classifier = TrendClassifier()
    print("✓ 走势识别器初始化")
    
    # 3. 背驰检测
    detector = DivergenceDetector()
    alpha = detector.calculate_divergence_alpha(df)
    df['alpha_divergence'] = alpha
    print(f"✓ 背驰因子计算完成, 均值={alpha.mean():.4f}")
    
    # 4. 区间套信号
    strategy = IntervalTrapStrategy()
    multi_level = {'day': df, '60m': df}
    signals = strategy.find_interval_trap_signals(multi_level)
    print(f"✓ 区间套信号: {len(signals)}个")
    
    print("\n所有P0模块集成测试完成!")

if __name__ == '__main__':
    print("="*60)
    print("P0增强模块集成演示")
    print("="*60)
    
    try:
        demo_p0_1_trend_classifier()
        demo_p0_2_divergence_detector()
        demo_p0_3_interval_trap()
        demo_p0_4_chart()
        demo_p0_6_backtest()
        demo_integration()
        
        print("\n" + "="*60)
        print("✅ 所有演示完成!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
