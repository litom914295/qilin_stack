"""
测试数据准备脚本 (Test Data Preparation)
Task 15: 自动化测试与口径校验

功能:
- 生成最小测试数据包 (日线/分钟线)
- 模拟行情数据 (用于 CI 环境)
- 一进二策略测试数据 (涨停/开板/一进二标的)
- IC 分析测试数据 (因子/标签)
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self, output_dir: str = "tests/data/qlib_data"):
        """
        初始化测试数据生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试标的池 (30只股票)
        self.stock_pool = [
            "000001.SZ", "000002.SZ", "000063.SZ", "000333.SZ", "000651.SZ",
            "000858.SZ", "002594.SZ", "002920.SZ", "300059.SZ", "300124.SZ",
            "600000.SH", "600036.SH", "600519.SH", "600887.SH", "601318.SH",
            "601888.SH", "603259.SH", "688005.SH", "688111.SH", "688599.SH",
            # 一进二测试专用标的 (包含涨停/开板/连板)
            "300750.SZ", "688981.SH", "000100.SZ", "600100.SH", "603100.SH",
            "002100.SZ", "688100.SH", "000200.SZ", "600200.SH", "603200.SH",
        ]
        
        # 时间范围 (2年历史数据)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=730)
    
    def generate_all(self):
        """生成所有测试数据"""
        print("开始生成测试数据...")
        
        # 1. 日线数据
        print("[1/6] 生成日线数据...")
        self.generate_daily_data()
        
        # 2. 分钟线数据 (最近30天)
        print("[2/6] 生成分钟线数据...")
        self.generate_minute_data()
        
        # 3. 一进二测试数据
        print("[3/6] 生成一进二测试数据...")
        self.generate_limitup_data()
        
        # 4. 因子数据 (用于 IC 分析)
        print("[4/6] 生成因子数据...")
        self.generate_factor_data()
        
        # 5. 模型训练数据
        print("[5/6] 生成模型训练数据...")
        self.generate_training_data()
        
        # 6. 回测数据
        print("[6/6] 生成回测数据...")
        self.generate_backtest_data()
        
        print(f"测试数据生成完成! 输出目录: {self.output_dir.absolute()}")
    
    def generate_daily_data(self):
        """生成日线数据"""
        daily_dir = self.output_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成交易日历
        trade_dates = pd.bdate_range(self.start_date, self.end_date, freq='B')
        
        for stock in self.stock_pool:
            # 基础价格 (根据股票代码生成不同基础价格)
            base_price = 10 + (hash(stock) % 100)
            
            # 生成价格序列 (几何布朗运动)
            returns = np.random.normal(0.0005, 0.02, len(trade_dates))
            prices = base_price * np.cumprod(1 + returns)
            
            # 生成 OHLCV 数据
            data = []
            for i, (date, close) in enumerate(zip(trade_dates, prices)):
                open_price = close * np.random.uniform(0.98, 1.02)
                high_price = max(open_price, close) * np.random.uniform(1.0, 1.05)
                low_price = min(open_price, close) * np.random.uniform(0.95, 1.0)
                volume = np.random.uniform(1e6, 1e8)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close, 2),
                    'volume': int(volume),
                    'amount': round(close * volume, 2),
                })
            
            df = pd.DataFrame(data)
            df.to_csv(daily_dir / f"{stock}.csv", index=False)
        
        print(f"  生成 {len(self.stock_pool)} 只股票的日线数据 ({len(trade_dates)} 天)")
    
    def generate_minute_data(self):
        """生成分钟线数据 (最近30天)"""
        minute_dir = self.output_dir / "1min"
        minute_dir.mkdir(parents=True, exist_ok=True)
        
        # 最近30个交易日
        trade_dates = pd.bdate_range(self.end_date - timedelta(days=45), self.end_date, freq='B')[-30:]
        
        # 交易时间段
        morning_hours = pd.date_range("09:30", "11:30", freq="1min").time
        afternoon_hours = pd.date_range("13:00", "15:00", freq="1min").time
        trading_minutes = list(morning_hours) + list(afternoon_hours)
        
        # 只为前10只股票生成分钟数据 (节省空间)
        for stock in self.stock_pool[:10]:
            base_price = 10 + (hash(stock) % 100)
            
            data = []
            for date in trade_dates:
                # 当日基准价格
                daily_base = base_price * np.random.uniform(0.9, 1.1)
                
                for minute_time in trading_minutes:
                    timestamp = datetime.combine(date.date(), minute_time)
                    
                    # 日内价格波动
                    close = daily_base * np.random.uniform(0.98, 1.02)
                    open_price = close * np.random.uniform(0.999, 1.001)
                    high_price = max(open_price, close) * np.random.uniform(1.0, 1.005)
                    low_price = min(open_price, close) * np.random.uniform(0.995, 1.0)
                    volume = np.random.uniform(1e4, 1e6)
                    
                    data.append({
                        'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close, 2),
                        'volume': int(volume),
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(minute_dir / f"{stock}.csv", index=False)
        
        print(f"  生成 {10} 只股票的分钟数据 ({len(trade_dates)} 天 x {len(trading_minutes)} 分钟)")
    
    def generate_limitup_data(self):
        """生成一进二测试数据 (涨停/开板/连板)"""
        limitup_dir = self.output_dir / "limitup"
        limitup_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成标签数据
        scenarios = [
            # 经典一进二: 低开<2%, 收盘涨停
            {
                'name': 'classic_yinjiner',
                'open_pct': -0.015,  # 低开 1.5%
                'close_pct': 0.0995,  # 收盘涨停 9.95%
                'next_day_pct': 0.03,  # 次日收益 3%
            },
            # 强势一进二: 高开>2%, 收盘涨停
            {
                'name': 'strong_yinjiner',
                'open_pct': 0.03,  # 高开 3%
                'close_pct': 0.0995,  # 收盘涨停
                'next_day_pct': 0.05,  # 次日收益 5%
            },
            # 连板: 一字涨停
            {
                'name': 'continuous_limitup',
                'open_pct': 0.0995,  # 开盘涨停
                'close_pct': 0.0995,  # 收盘涨停
                'next_day_pct': 0.0995,  # 次日继续涨停
            },
            # 开板反包: 盘中开板后重新封板
            {
                'name': 'open_board_reverse',
                'open_pct': 0.05,
                'close_pct': 0.0995,
                'next_day_pct': 0.02,
            },
            # 普通上涨 (对照组)
            {
                'name': 'normal_up',
                'open_pct': 0.0,
                'close_pct': 0.03,
                'next_day_pct': 0.01,
            },
        ]
        
        trade_dates = pd.bdate_range(self.end_date - timedelta(days=365), self.end_date, freq='B')
        
        for i, scenario in enumerate(scenarios):
            stock = self.stock_pool[20 + i]  # 使用特定股票
            base_price = 20.0
            
            data = []
            for j, date in enumerate(trade_dates[:-1]):  # 留一天计算次日收益
                prev_close = base_price * (1 + np.random.normal(0, 0.01))
                
                # 模拟场景
                if j % 10 == 0:  # 每10天出现一次目标场景
                    open_price = prev_close * (1 + scenario['open_pct'])
                    close_price = prev_close * (1 + scenario['close_pct'])
                    next_close = close_price * (1 + scenario['next_day_pct'])
                else:  # 其他时间正常波动
                    open_price = prev_close * np.random.uniform(0.98, 1.02)
                    close_price = prev_close * np.random.uniform(0.97, 1.03)
                    next_close = close_price * np.random.uniform(0.98, 1.02)
                
                high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.01)
                low_price = min(open_price, close_price) * np.random.uniform(0.99, 1.0)
                volume = np.random.uniform(5e6, 5e7)
                
                # 标签: 次日收益
                label = (next_close - close_price) / close_price
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume),
                    'label_return': round(label, 4),
                    'is_limitup': int(abs(close_price/prev_close - 1.0995) < 0.001),
                    'scenario': scenario['name'],
                })
                
                base_price = close_price
            
            df = pd.DataFrame(data)
            df.to_csv(limitup_dir / f"{stock}_{scenario['name']}.csv", index=False)
        
        print(f"  生成 {len(scenarios)} 种一进二场景的测试数据")
    
    def generate_factor_data(self):
        """生成因子数据 (用于 IC 分析)"""
        factor_dir = self.output_dir / "factors"
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        trade_dates = pd.bdate_range(self.end_date - timedelta(days=365), self.end_date, freq='B')
        
        data = []
        for stock in self.stock_pool:
            for date in trade_dates:
                # 生成多个因子
                factors = {
                    'date': date.strftime('%Y-%m-%d'),
                    'instrument': stock,
                    # 价格因子
                    'factor_price_momentum': np.random.normal(0, 1),
                    'factor_price_reversal': np.random.normal(0, 1),
                    # 成交量因子
                    'factor_volume_ratio': np.random.uniform(0.5, 2.0),
                    'factor_turnover': np.random.uniform(0.01, 0.1),
                    # 技术因子
                    'factor_rsi': np.random.uniform(20, 80),
                    'factor_macd': np.random.normal(0, 0.5),
                    # 标签 (未来5日收益)
                    'label_5d_return': np.random.normal(0.01, 0.03),
                }
                data.append(factors)
        
        df = pd.DataFrame(data)
        df.to_csv(factor_dir / "factors.csv", index=False)
        
        print(f"  生成 {len(self.stock_pool)} x {len(trade_dates)} 条因子数据")
    
    def generate_training_data(self):
        """生成模型训练数据"""
        training_dir = self.output_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成特征矩阵和标签
        trade_dates = pd.bdate_range(self.end_date - timedelta(days=730), self.end_date, freq='B')
        
        data = []
        for stock in self.stock_pool:
            for date in trade_dates:
                features = {
                    'date': date.strftime('%Y-%m-%d'),
                    'instrument': stock,
                }
                
                # 生成 20 个随机特征 (模拟 Alpha158 的一部分)
                for i in range(20):
                    features[f'feature_{i:02d}'] = np.random.normal(0, 1)
                
                # 标签 (未来1日收益)
                features['label'] = np.random.normal(0.001, 0.02)
                
                data.append(features)
        
        df = pd.DataFrame(data)
        df.to_csv(training_dir / "training_data.csv", index=False)
        
        print(f"  生成模型训练数据: {len(data)} 样本")
    
    def generate_backtest_data(self):
        """生成回测数据 (包含预测/实际收益)"""
        backtest_dir = self.output_dir / "backtest"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        trade_dates = pd.bdate_range(self.end_date - timedelta(days=180), self.end_date, freq='B')
        
        data = []
        for stock in self.stock_pool[:20]:  # 只用 20 只股票
            for date in trade_dates:
                # 模拟预测分数与实际收益
                pred_score = np.random.normal(0, 1)
                actual_return = pred_score * 0.01 + np.random.normal(0, 0.02)  # 有一定相关性
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'instrument': stock,
                    'score': round(pred_score, 4),
                    'return': round(actual_return, 4),
                })
        
        df = pd.DataFrame(data)
        df.to_csv(backtest_dir / "predictions.csv", index=False)
        
        print(f"  生成回测数据: {len(data)} 条预测")
    
    def generate_data_meta(self):
        """生成数据元信息"""
        meta = {
            'version': '1.0.0',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock_pool': self.stock_pool,
            'date_range': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d'),
            },
            'data_types': [
                'daily',       # 日线数据
                '1min',        # 分钟数据
                'limitup',     # 一进二数据
                'factors',     # 因子数据
                'training',    # 训练数据
                'backtest',    # 回测数据
            ],
            'usage': {
                'daily': '用于日线回测、因子计算',
                '1min': '用于高频交易、NestedExecutor',
                'limitup': '用于一进二策略测试',
                'factors': '用于 IC 分析测试',
                'training': '用于模型训练测试',
                'backtest': '用于回测结果校验',
            }
        }
        
        import json
        with open(self.output_dir / "data_meta.json", 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"  生成数据元信息: data_meta.json")


def main():
    """主函数"""
    generator = TestDataGenerator()
    generator.generate_all()
    generator.generate_data_meta()
    
    print("\n测试数据生成完成! 可用于:")
    print("  - 单元测试: tests/unit/")
    print("  - 集成测试: tests/integration/")
    print("  - E2E 测试: tests/e2e/")
    print("  - CI 自动化测试")


if __name__ == "__main__":
    main()
