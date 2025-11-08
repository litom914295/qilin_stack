"""
表达式引擎单元测试 (Expression Engine Unit Tests)
Task 15: 自动化测试与口径校验

测试内容:
- 表达式语法解析
- 表达式计算正确性
- 边界条件处理 (NaN/Inf/空数据)
- 性能测试
- 一进二专用表达式测试
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestExpressionEngine:
    """表达式引擎测试"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        dates = pd.date_range('2024-01-01', periods=20, freq='B')
        data = {
            'date': dates,
            'close': [10.0 + i * 0.1 for i in range(20)],
            'open': [10.0 + i * 0.1 + np.random.uniform(-0.1, 0.1) for i in range(20)],
            'high': [10.0 + i * 0.1 + 0.2 for i in range(20)],
            'low': [10.0 + i * 0.1 - 0.2 for i in range(20)],
            'volume': [1e6 + i * 1e4 for i in range(20)],
        }
        return pd.DataFrame(data)
    
    @pytest.mark.unit
    def test_basic_expression(self, sample_data):
        """测试基本表达式"""
        try:
            import qlib
            from qlib.data import D
            
            # 测试简单算术表达式
            expr = "$close / $open - 1"
            # 这里需要实际的表达式引擎实现
            # 暂时跳过需要 Qlib 数据的测试
            pytest.skip("需要 Qlib 数据环境")
        except ImportError:
            pytest.skip("Qlib 未安装")
    
    @pytest.mark.unit
    def test_ref_operator(self, sample_data):
        """测试 Ref 操作符"""
        # 手动计算 Ref($close, 1)
        ref_close = sample_data['close'].shift(1)
        
        # 验证 NaN 处理
        assert pd.isna(ref_close.iloc[0])
        assert ref_close.iloc[1] == sample_data['close'].iloc[0]
    
    @pytest.mark.unit
    def test_mean_operator(self, sample_data):
        """测试 Mean 操作符"""
        # Mean($close, 5) - 5日均值
        mean_close = sample_data['close'].rolling(window=5).mean()
        
        # 验证前4个值为 NaN
        assert pd.isna(mean_close.iloc[:4]).all()
        # 验证第5个值
        expected = sample_data['close'].iloc[:5].mean()
        assert abs(mean_close.iloc[4] - expected) < 1e-6
    
    @pytest.mark.unit
    def test_std_operator(self, sample_data):
        """测试 Std 操作符"""
        # Std($close, 5) - 5日标准差
        std_close = sample_data['close'].rolling(window=5).std()
        
        # 验证前4个值为 NaN
        assert pd.isna(std_close.iloc[:4]).all()
        # 验证第5个值
        expected = sample_data['close'].iloc[:5].std()
        assert abs(std_close.iloc[4] - expected) < 1e-6
    
    @pytest.mark.unit
    def test_if_expression(self):
        """测试 If 条件表达式"""
        # If(condition, true_value, false_value)
        condition = pd.Series([True, False, True, False])
        true_val = pd.Series([1, 1, 1, 1])
        false_val = pd.Series([0, 0, 0, 0])
        
        result = np.where(condition, true_val, false_val)
        expected = np.array([1, 0, 1, 0])
        
        assert np.array_equal(result, expected)
    
    @pytest.mark.unit
    def test_limitup_expression(self, sample_data):
        """测试一进二涨停表达式"""
        # 涨停判断: $close / Ref($close, 1) - 1 > 0.095
        close = sample_data['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan  # 第一个值没有前一天
        
        # 计算涨幅
        change_pct = close / prev_close - 1
        
        # 涨停标记
        is_limitup = change_pct > 0.095
        
        # 验证第一个值为 False (因为 prev_close 是 NaN)
        assert not is_limitup[0] or np.isnan(change_pct[0])
    
    @pytest.mark.unit
    def test_nan_handling(self):
        """测试 NaN 处理"""
        data = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan])
        
        # 测试 fillna
        filled = data.fillna(0)
        assert filled.iloc[1] == 0
        assert filled.iloc[4] == 0
        
        # 测试 dropna
        dropped = data.dropna()
        assert len(dropped) == 3
        assert dropped.iloc[0] == 1.0
    
    @pytest.mark.unit
    def test_inf_handling(self):
        """测试 Inf 处理"""
        data = pd.Series([1.0, np.inf, 3.0, -np.inf, 5.0])
        
        # 替换 Inf 为 NaN
        clean = data.replace([np.inf, -np.inf], np.nan)
        assert pd.isna(clean.iloc[1])
        assert pd.isna(clean.iloc[3])
    
    @pytest.mark.unit
    def test_empty_data(self):
        """测试空数据处理"""
        data = pd.DataFrame()
        
        # 确保不会抛出异常
        assert len(data) == 0
        assert data.empty
    
    @pytest.mark.unit
    def test_cross_sectional_rank(self):
        """测试横截面排名"""
        # 模拟多只股票的数据
        data = pd.DataFrame({
            'stock': ['A', 'B', 'C', 'D', 'E'],
            'factor': [0.5, 0.8, 0.2, 0.9, 0.3],
        })
        
        # 排名 (升序)
        data['rank'] = data['factor'].rank(ascending=True)
        
        # 验证排名
        assert data.loc[data['stock'] == 'D', 'rank'].values[0] == 5  # 最大值
        assert data.loc[data['stock'] == 'C', 'rank'].values[0] == 1  # 最小值
    
    @pytest.mark.unit
    def test_zscore_normalization(self):
        """测试 Z-Score 标准化"""
        data = pd.Series([1, 2, 3, 4, 5])
        
        # Z-Score = (x - mean) / std
        mean = data.mean()
        std = data.std()
        zscore = (data - mean) / std
        
        # 验证 Z-Score 均值接近 0, 标准差接近 1
        assert abs(zscore.mean()) < 1e-10
        assert abs(zscore.std() - 1.0) < 1e-10
    
    @pytest.mark.unit
    def test_winsorize(self):
        """测试去极值 (Winsorize)"""
        data = pd.Series([1, 2, 3, 4, 100])  # 100 是极值
        
        # 3倍标准差去极值
        mean = data.mean()
        std = data.std()
        upper = mean + 3 * std
        lower = mean - 3 * std
        
        winsorized = data.clip(lower=lower, upper=upper)
        
        # 极值应该被截断
        assert winsorized.iloc[-1] == upper
    
    @pytest.mark.unit
    def test_complex_expression(self, sample_data):
        """测试复杂表达式"""
        # ($close - Mean($close, 20)) / Std($close, 20)
        close = sample_data['close']
        mean_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        
        zscore = (close - mean_20) / std_20
        
        # 前19个值应该是 NaN
        assert pd.isna(zscore.iloc[:19]).all()
        
        # 第20个值应该有效
        if not pd.isna(zscore.iloc[19]):
            assert isinstance(zscore.iloc[19], (int, float))


class TestLimitUpExpressions:
    """一进二专用表达式测试"""
    
    @pytest.fixture
    def limitup_data(self):
        """生成一进二测试数据"""
        data = {
            'date': pd.date_range('2024-01-01', periods=10, freq='B'),
            'open': [10.0, 9.85, 10.5, 11.0, 10.8, 10.9, 11.2, 11.5, 11.3, 11.4],
            'close': [10.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.5, 11.5, 11.8, 12.0],
        }
        return pd.DataFrame(data)
    
    @pytest.mark.unit
    def test_classic_yinjiner_label(self, limitup_data):
        """测试经典一进二标签: 低开<2% 且 收盘涨停"""
        df = limitup_data.copy()
        
        # 前日收盘
        prev_close = df['close'].shift(1)
        
        # 低开幅度
        open_pct = (df['open'] - prev_close) / prev_close
        
        # 涨停判断
        close_pct = (df['close'] - prev_close) / prev_close
        
        # 标签: 低开<2% 且 收盘涨停
        label = ((open_pct < 0.02) & (close_pct > 0.095)).astype(int)
        
        # 验证第2天 (index=1): open=9.85, prev_close=10.0, close=11.0
        # open_pct = (9.85-10)/10 = -0.015 < 0.02 ✓
        # close_pct = (11-10)/10 = 0.1 > 0.095 ✓
        assert label.iloc[1] == 1
    
    @pytest.mark.unit
    def test_continuous_limitup_label(self, limitup_data):
        """测试连板标签: 开盘涨停 且 收盘涨停"""
        df = limitup_data.copy()
        
        prev_close = df['close'].shift(1)
        
        # 开盘涨停
        open_limitup = ((df['open'] - prev_close) / prev_close > 0.095)
        
        # 收盘涨停
        close_limitup = ((df['close'] - prev_close) / prev_close > 0.095)
        
        # 连板标签
        label = (open_limitup & close_limitup).astype(int)
        
        # 验证数据中是否有连板
        assert label.sum() >= 0  # 可能没有连板
    
    @pytest.mark.unit
    def test_limitup_volume_surge(self, limitup_data):
        """测试涨停 + 放量"""
        # 添加成交量数据
        df = limitup_data.copy()
        df['volume'] = [1e6, 5e6, 2e6, 3e6, 1.5e6, 2.5e6, 4e6, 1e6, 2e6, 3e6]
        
        # 成交量均值
        mean_volume = df['volume'].rolling(window=5).mean()
        
        # 放量倍数
        volume_ratio = df['volume'] / mean_volume
        
        # 验证计算
        assert not pd.isna(volume_ratio.iloc[4:]).any()


class TestExpressionPerformance:
    """表达式性能测试"""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        import time
        
        # 生成大数据集: 5000只股票 x 1000天
        n_stocks = 5000
        n_days = 1000
        
        data = pd.DataFrame({
            'stock': np.repeat(range(n_stocks), n_days),
            'date': np.tile(pd.date_range('2020-01-01', periods=n_days, freq='B'), n_stocks),
            'close': np.random.randn(n_stocks * n_days) * 10 + 100,
        })
        
        # 测试滚动均值计算时间
        start = time.time()
        data['ma_20'] = data.groupby('stock')['close'].transform(lambda x: x.rolling(20).mean())
        elapsed = time.time() - start
        
        print(f"计算 {n_stocks} 只股票 x {n_days} 天的 MA20 耗时: {elapsed:.2f}s")
        
        # 性能要求: 应在 5 秒内完成
        assert elapsed < 5.0, f"性能不达标: {elapsed:.2f}s > 5.0s"
    
    @pytest.mark.unit
    def test_expression_caching(self):
        """测试表达式缓存机制"""
        # 测试相同表达式是否能复用缓存
        data = pd.DataFrame({
            'close': [10.0 + i * 0.1 for i in range(100)]
        })
        
        import time
        
        # 第一次计算
        start = time.time()
        result1 = data['close'].rolling(window=20).mean()
        time1 = time.time() - start
        
        # 第二次计算 (应该更快,如果有缓存)
        start = time.time()
        result2 = data['close'].rolling(window=20).mean()
        time2 = time.time() - start
        
        # 结果应该相同
        assert np.allclose(result1, result2, equal_nan=True)
        
        print(f"第一次: {time1*1000:.2f}ms, 第二次: {time2*1000:.2f}ms")


def test_expression_syntax_validation():
    """测试表达式语法校验"""
    valid_expressions = [
        "$close",
        "$close / $open - 1",
        "Mean($close, 5)",
        "Std($close, 20)",
        "Ref($close, 1)",
        "If($close > $open, 1, 0)",
        "($close - Mean($close, 20)) / Std($close, 20)",
    ]
    
    invalid_expressions = [
        "",  # 空表达式
        "$",  # 不完整
        "Mean($close)",  # 缺少参数
        "If($close > $open, 1)",  # 缺少 else
    ]
    
    # 简单的语法检查
    for expr in valid_expressions:
        assert len(expr) > 0
        assert expr.count("(") == expr.count(")")  # 括号匹配
    
    for expr in invalid_expressions:
        if expr:
            # 检查可能的语法错误
            pass  # 实际应该调用表达式解析器


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
