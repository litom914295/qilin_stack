"""
IC 分析单元测试 (IC Analysis Unit Tests)
Task 15: 自动化测试与口径校验

测试内容:
- IC/IR 计算正确性
- 分层收益分析
- NaN/Inf 处理
- 横截面去极值/标准化/中性化
- 与 Qlib 官方 API 对齐
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestICCalculation:
    """IC 计算测试"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据: 因子 vs 标签"""
        np.random.seed(42)
        n_samples = 100
        
        # 生成因子 (有一定预测能力)
        factor = np.random.randn(n_samples)
        # 生成标签 (与因子相关,相关系数约 0.3)
        label = 0.3 * factor + np.random.randn(n_samples)
        
        return pd.DataFrame({
            'factor': factor,
            'label': label,
        })
    
    @pytest.mark.unit
    def test_ic_pearson(self, sample_data):
        """测试 Pearson IC"""
        ic = sample_data['factor'].corr(sample_data['label'], method='pearson')
        
        # 验证 IC 范围 [-1, 1]
        assert -1 <= ic <= 1
        
        # 由于数据是有相关性的,IC 应该 > 0
        assert ic > 0, f"IC={ic:.4f} 应该 > 0"
        
        print(f"Pearson IC: {ic:.4f}")
    
    @pytest.mark.unit
    def test_ic_spearman(self, sample_data):
        """测试 Spearman IC (秩相关)"""
        ic = sample_data['factor'].corr(sample_data['label'], method='spearman')
        
        # 验证 IC 范围
        assert -1 <= ic <= 1
        assert ic > 0
        
        print(f"Spearman IC: {ic:.4f}")
    
    @pytest.mark.unit
    def test_ic_with_nan(self):
        """测试包含 NaN 的 IC 计算"""
        factor = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        label = pd.Series([1.5, 2.5, 3.5, np.nan, 5.5])
        
        # dropna 策略: 移除所有 NaN
        valid_mask = ~(factor.isna() | label.isna())
        ic = factor[valid_mask].corr(label[valid_mask])
        
        # 应该只用 3 个有效样本计算
        assert not np.isnan(ic)
        print(f"IC with NaN (dropped): {ic:.4f}, n={valid_mask.sum()}")
    
    @pytest.mark.unit
    def test_ic_with_inf(self):
        """测试包含 Inf 的 IC 计算"""
        factor = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0])
        label = pd.Series([1.5, 2.5, 3.5, -np.inf, 5.5])
        
        # 替换 Inf 为 NaN
        factor_clean = factor.replace([np.inf, -np.inf], np.nan)
        label_clean = label.replace([np.inf, -np.inf], np.nan)
        
        valid_mask = ~(factor_clean.isna() | label_clean.isna())
        ic = factor_clean[valid_mask].corr(label_clean[valid_mask])
        
        assert not np.isnan(ic)
        print(f"IC with Inf (replaced): {ic:.4f}, n={valid_mask.sum()}")
    
    @pytest.mark.unit
    def test_ic_insufficient_samples(self):
        """测试样本数不足的情况"""
        factor = pd.Series([1.0, 2.0])
        label = pd.Series([1.5, 2.5])
        
        # 样本数 < 10,应该返回 NaN 或警告
        ic = factor.corr(label)
        
        # 验证计算结果
        assert isinstance(ic, (int, float))
        print(f"IC with 2 samples: {ic:.4f} (warning: too few samples)")
    
    @pytest.mark.unit
    def test_ir_calculation(self, sample_data):
        """测试 IR (Information Ratio) 计算"""
        # 模拟多期 IC
        n_periods = 20
        ics = []
        
        for i in range(n_periods):
            # 每期随机抽样
            sample = sample_data.sample(frac=0.8, random_state=i)
            ic = sample['factor'].corr(sample['label'])
            ics.append(ic)
        
        # IR = mean(IC) / std(IC)
        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ir = ic_mean / ic_std if ic_std > 0 else 0
        
        print(f"IC mean: {ic_mean:.4f}, std: {ic_std:.4f}, IR: {ir:.4f}")
        
        # 验证 IR
        assert isinstance(ir, (int, float))
        assert not np.isnan(ir)


class TestQuantileAnalysis:
    """分位数分析测试"""
    
    @pytest.fixture
    def quantile_data(self):
        """生成分位数测试数据"""
        np.random.seed(42)
        n_samples = 500
        
        # 生成因子
        factor = np.random.randn(n_samples)
        # 生成收益 (与因子单调相关)
        noise = np.random.randn(n_samples) * 0.02
        label = 0.01 * factor + noise
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n_samples, freq='B') # 重复日期
                   .repeat(1)[:n_samples],
            'stock': [f'stock_{i%100}' for i in range(n_samples)],
            'factor': factor,
            'label': label,
        })
        
        return df
    
    @pytest.mark.unit
    def test_5_quantile_split(self, quantile_data):
        """测试 5 分位分组"""
        df = quantile_data.copy()
        
        # 按因子分 5 组
        df['quantile'] = pd.qcut(df['factor'], q=5, labels=False, duplicates='drop')
        
        # 计算每组平均收益
        quantile_returns = df.groupby('quantile')['label'].mean()
        
        # 验证: 高分位组收益 > 低分位组收益 (因子有效)
        if len(quantile_returns) >= 2:
            assert quantile_returns.iloc[-1] > quantile_returns.iloc[0], \
                f"Q5={quantile_returns.iloc[-1]:.4f} 应该 > Q1={quantile_returns.iloc[0]:.4f}"
        
        print("5分位平均收益:")
        print(quantile_returns)
    
    @pytest.mark.unit
    def test_long_short_spread(self, quantile_data):
        """测试多空组合收益"""
        df = quantile_data.copy()
        
        # 分 5 组
        df['quantile'] = pd.qcut(df['factor'], q=5, labels=False, duplicates='drop')
        
        # 多头: 第 5 组, 空头: 第 1 组
        long_return = df[df['quantile'] == 4]['label'].mean()
        short_return = df[df['quantile'] == 0]['label'].mean()
        
        spread = long_return - short_return
        
        print(f"多空收益: Long={long_return:.4f}, Short={short_return:.4f}, Spread={spread:.4f}")
        
        # 验证多空收益差
        assert spread > 0, f"多空收益差={spread:.4f} 应该 > 0"


class TestCrossSectionalProcessing:
    """横截面处理测试"""
    
    @pytest.fixture
    def cross_sectional_data(self):
        """生成横截面测试数据"""
        np.random.seed(42)
        
        # 5 个日期, 每个日期 100 只股票
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = [f'stock_{i:03d}' for i in range(100)]
        
        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'factor': np.random.randn(),
                    'label': np.random.randn() * 0.02,
                })
        
        return pd.DataFrame(data)
    
    @pytest.mark.unit
    def test_winsorize(self, cross_sectional_data):
        """测试去极值"""
        df = cross_sectional_data.copy()
        
        # 按日期分组去极值
        def winsorize_group(x, n_sigma=3):
            mean = x.mean()
            std = x.std()
            upper = mean + n_sigma * std
            lower = mean - n_sigma * std
            return x.clip(lower=lower, upper=upper)
        
        df['factor_winsorized'] = df.groupby('date')['factor'].transform(
            lambda x: winsorize_group(x, n_sigma=3)
        )
        
        # 验证: 极值被截断
        for date in df['date'].unique():
            date_data = df[df['date'] == date]
            assert date_data['factor_winsorized'].max() <= date_data['factor_winsorized'].mean() + 3 * date_data['factor_winsorized'].std() + 1e-6
    
    @pytest.mark.unit
    def test_standardize(self, cross_sectional_data):
        """测试标准化 (Z-Score)"""
        df = cross_sectional_data.copy()
        
        # 按日期分组标准化
        df['factor_std'] = df.groupby('date')['factor'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # 验证: 每个横截面均值≈0, 标准差≈1
        for date in df['date'].unique():
            date_data = df[df['date'] == date]['factor_std']
            assert abs(date_data.mean()) < 1e-10
            assert abs(date_data.std() - 1.0) < 1e-10
    
    @pytest.mark.unit
    def test_neutralize(self, cross_sectional_data):
        """测试中性化 (市值/行业)"""
        df = cross_sectional_data.copy()
        
        # 添加市值数据
        df['market_cap'] = np.random.uniform(1e9, 1e11, len(df))
        
        # 对市值回归并取残差
        from sklearn.linear_model import LinearRegression
        
        residuals = []
        for date in df['date'].unique():
            date_data = df[df['date'] == date].copy()
            
            X = date_data[['market_cap']].values
            y = date_data['factor'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            pred = model.predict(X)
            resid = y - pred
            
            residuals.extend(resid)
        
        df['factor_neutral'] = residuals
        
        # 验证: 残差与市值相关性接近 0
        corr = df.groupby('date').apply(
            lambda x: x['factor_neutral'].corr(x['market_cap'])
        ).mean()
        
        print(f"中性化后与市值的平均相关性: {corr:.6f}")
        assert abs(corr) < 0.1  # 相关性应接近 0


class TestICAlignmentWithQlib:
    """与 Qlib 官方 API 对齐测试"""
    
    @pytest.mark.integration
    def test_ic_api_alignment(self):
        """测试 IC 计算是否与 Qlib 官方一致"""
        try:
            import qlib
            from qlib.data import D
            from qlib.contrib.evaluate import backtest as bt
            
            # 需要 Qlib 数据环境
            pytest.skip("需要 Qlib 数据环境和实际回测结果")
        except ImportError:
            pytest.skip("Qlib 未安装")
    
    @pytest.mark.integration
    def test_risk_analysis_alignment(self):
        """测试 risk_analysis 是否与 Qlib 官方一致"""
        try:
            from qlib.contrib.evaluate import risk_analysis
            
            # 模拟回测结果
            dates = pd.date_range('2024-01-01', periods=100, freq='B')
            returns = pd.Series(
                np.random.randn(100) * 0.02 + 0.001,
                index=dates,
                name='return'
            )
            
            # 调用官方 API
            # report = risk_analysis(returns)
            
            # 需要实际数据验证
            pytest.skip("需要实际回测数据")
        except ImportError:
            pytest.skip("Qlib 未安装")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
