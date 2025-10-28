"""
综合系统实现
包含：风险管理系统、回测引擎、绩效分析系统、数据管理系统、部署配置
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging
from pathlib import Path
import json
import yaml
import pickle
import sqlite3
from scipy import stats
import docker
import redis
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


# ==================== 风险管理系统 ====================

class RiskManagementSystem:
    """风险管理系统 - VaR计算、压力测试、风险预警"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.var_confidence = config.get('var_confidence', 0.95)
        self.stress_scenarios = config.get('stress_scenarios', [])
        self.risk_limits = config.get('risk_limits', {})
        self.alerts = []
        
    def calculate_var(self, returns: pd.Series, method: str = 'historical') -> float:
        """计算VaR（在险价值）"""
        if method == 'historical':
            return self._historical_var(returns)
        elif method == 'parametric':
            return self._parametric_var(returns)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(returns)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, returns: pd.Series) -> float:
        """历史模拟法VaR"""
        return returns.quantile(1 - self.var_confidence)
    
    def _parametric_var(self, returns: pd.Series) -> float:
        """参数法VaR"""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - self.var_confidence)
        return mean + z_score * std
    
    def _monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000) -> float:
        """蒙特卡洛法VaR"""
        mean = returns.mean()
        std = returns.std()
        simulated_returns = np.random.normal(mean, std, n_simulations)
        return np.percentile(simulated_returns, (1 - self.var_confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series) -> float:
        """计算CVaR（条件在险价值）"""
        var = self.calculate_var(returns)
        return returns[returns <= var].mean()
    
    def stress_test(self, portfolio: pd.DataFrame, scenarios: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """压力测试"""
        if scenarios is None:
            scenarios = self.stress_scenarios or self._default_scenarios()
        
        results = {}
        for scenario in scenarios:
            scenario_name = scenario['name']
            shock = scenario['shock']
            
            # 应用冲击
            stressed_portfolio = portfolio.copy()
            for asset, factor in shock.items():
                if asset in stressed_portfolio.columns:
                    stressed_portfolio[asset] *= (1 + factor)
            
            # 计算损失
            original_value = portfolio.sum().sum()
            stressed_value = stressed_portfolio.sum().sum()
            loss = original_value - stressed_value
            loss_pct = loss / original_value
            
            results[scenario_name] = {
                'loss': loss,
                'loss_pct': loss_pct,
                'stressed_value': stressed_value
            }
        
        return results
    
    def _default_scenarios(self) -> List[Dict]:
        """默认压力测试场景"""
        return [
            {
                'name': '市场崩盘',
                'shock': {'market': -0.20}
            },
            {
                'name': '流动性危机',
                'shock': {'liquidity': -0.15, 'spread': 0.05}
            },
            {
                'name': '黑天鹅事件',
                'shock': {'market': -0.30, 'volatility': 0.50}
            }
        ]
    
    def check_risk_limits(self, metrics: Dict[str, float]) -> List[str]:
        """检查风险限制"""
        violations = []
        
        for metric, value in metrics.items():
            if metric in self.risk_limits:
                limit = self.risk_limits[metric]
                if value > limit:
                    violations.append(f"{metric} exceeds limit: {value:.4f} > {limit:.4f}")
        
        return violations
    
    def generate_risk_alert(self, message: str, severity: str = 'INFO'):
        """生成风险预警"""
        alert = {
            'timestamp': datetime.now(),
            'severity': severity,
            'message': message
        }
        self.alerts.append(alert)
        logger.log(getattr(logging, severity), f"Risk Alert: {message}")
        
        return alert
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算综合风险指标"""
        return {
            'var_95': self.calculate_var(returns, 'historical'),
            'cvar_95': self.calculate_cvar(returns),
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'downside_deviation': self._calculate_downside_deviation(returns),
            'beta': self._calculate_beta(returns)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_downside_deviation(self, returns: pd.Series, mar: float = 0) -> float:
        """计算下行偏差"""
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return 0
        return np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
    
    def _calculate_beta(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> float:
        """计算Beta系数"""
        if market_returns is None:
            return 1.0
        
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 0
        
        return covariance / market_variance


# ==================== 回测引擎 ====================

class BacktestEngine:
    """回测引擎 - 历史数据回测、绩效分析、交易成本模拟"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_capital = config.get('initial_capital', 1000000)
        self.commission_rate = config.get('commission_rate', 0.0003)
        self.slippage = config.get('slippage', 0.0001)
        self.results = {}
        
    async def run_backtest(self, 
                          strategy,
                          data: pd.DataFrame,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """运行回测"""
        # 过滤数据
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 初始化
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'value': self.initial_capital,
            'returns': [],
            'trades': []
        }
        
        # 逐日回测
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            
            # 生成信号
            signal = await strategy.generate_signal(current_data)
            
            # 执行交易
            if signal['signal'] == 'BUY':
                self._execute_buy(portfolio, current_price, signal)
            elif signal['signal'] == 'SELL':
                self._execute_sell(portfolio, current_price, signal)
            
            # 更新组合价值
            self._update_portfolio_value(portfolio, current_price)
            
            # 记录收益
            if i > 0:
                prev_value = self.results.get('portfolio_values', [self.initial_capital])[-1]
                daily_return = (portfolio['value'] - prev_value) / prev_value
                portfolio['returns'].append(daily_return)
        
        # 计算绩效
        self.results = self._calculate_performance(portfolio, data)
        
        return self.results
    
    def _execute_buy(self, portfolio: Dict, price: float, signal: Dict):
        """执行买入"""
        # 计算可买数量
        available_cash = portfolio['cash'] * 0.95  # 保留5%现金
        shares = int(available_cash / price / 100) * 100  # 整手
        
        if shares > 0:
            cost = shares * price * (1 + self.commission_rate + self.slippage)
            
            if cost <= portfolio['cash']:
                portfolio['cash'] -= cost
                portfolio['positions'][signal.get('symbol', 'default')] = {
                    'shares': shares,
                    'avg_price': price * (1 + self.slippage)
                }
                
                portfolio['trades'].append({
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'cost': cost,
                    'timestamp': datetime.now()
                })
    
    def _execute_sell(self, portfolio: Dict, price: float, signal: Dict):
        """执行卖出"""
        symbol = signal.get('symbol', 'default')
        
        if symbol in portfolio['positions']:
            position = portfolio['positions'][symbol]
            shares = position['shares']
            
            proceeds = shares * price * (1 - self.commission_rate - self.slippage)
            portfolio['cash'] += proceeds
            
            # 记录交易
            portfolio['trades'].append({
                'type': 'SELL',
                'price': price,
                'shares': shares,
                'proceeds': proceeds,
                'pnl': proceeds - shares * position['avg_price'],
                'timestamp': datetime.now()
            })
            
            # 清除持仓
            del portfolio['positions'][symbol]
    
    def _update_portfolio_value(self, portfolio: Dict, current_price: float):
        """更新组合价值"""
        positions_value = sum(
            pos['shares'] * current_price 
            for pos in portfolio['positions'].values()
        )
        portfolio['value'] = portfolio['cash'] + positions_value
    
    def _calculate_performance(self, portfolio: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """计算绩效指标"""
        returns = pd.Series(portfolio['returns'])
        
        if len(returns) == 0:
            return {}
        
        # 基础指标
        total_return = (portfolio['value'] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 风险调整指标
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, annual_return)
        
        # 交易统计
        trades = portfolio['trades']
        num_trades = len(trades)
        win_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': self._calculate_max_drawdown_from_returns(returns),
            'num_trades': num_trades,
            'win_rate': win_rate,
            'portfolio_value': portfolio['value'],
            'final_cash': portfolio['cash'],
            'trades': trades
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series, mar: float = 0) -> float:
        """计算Sortino比率"""
        excess_returns = returns - mar
        downside_returns = returns[returns < mar]
        
        if len(downside_returns) == 0:
            return 0
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        if downside_std == 0:
            return 0
        
        return excess_returns.mean() * 252 / downside_std
    
    def _calculate_calmar_ratio(self, returns: pd.Series, annual_return: float) -> float:
        """计算Calmar比率"""
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        
        if max_dd == 0:
            return 0
        
        return annual_return / max_dd
    
    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """从收益率计算最大回撤"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def generate_report(self) -> str:
        """生成回测报告"""
        if not self.results:
            return "No backtest results available"
        
        report = []
        report.append("=" * 50)
        report.append("BACKTEST REPORT")
        report.append("=" * 50)
        report.append(f"Total Return: {self.results['total_return']:.2%}")
        report.append(f"Annual Return: {self.results['annual_return']:.2%}")
        report.append(f"Volatility: {self.results['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {self.results['max_drawdown']:.2%}")
        report.append(f"Number of Trades: {self.results['num_trades']}")
        report.append(f"Win Rate: {self.results['win_rate']:.2%}")
        report.append("=" * 50)
        
        return "\n".join(report)


# ==================== 绩效分析系统 ====================

class PerformanceAnalyzer:
    """绩效分析系统 - 收益分析、风险调整收益、归因分析"""
    
    def __init__(self):
        self.metrics = {}
        
    def analyze_returns(self, returns: pd.Series) -> Dict[str, Any]:
        """分析收益"""
        return {
            'total_return': self._total_return(returns),
            'annual_return': self._annualized_return(returns),
            'monthly_returns': self._monthly_returns(returns),
            'rolling_returns': self._rolling_returns(returns),
            'return_distribution': self._return_distribution(returns)
        }
    
    def _total_return(self, returns: pd.Series) -> float:
        """计算总收益"""
        return (1 + returns).prod() - 1
    
    def _annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益"""
        total_return = self._total_return(returns)
        years = len(returns) / 252
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    def _monthly_returns(self, returns: pd.Series) -> pd.Series:
        """计算月度收益"""
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            return monthly
        return pd.Series()
    
    def _rolling_returns(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """计算滚动收益"""
        return returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
    
    def _return_distribution(self, returns: pd.Series) -> Dict[str, float]:
        """收益分布分析"""
        return {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min': returns.min(),
            'max': returns.max(),
            'percentile_5': returns.quantile(0.05),
            'percentile_95': returns.quantile(0.95)
        }
    
    def calculate_risk_adjusted_returns(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
        """计算风险调整收益"""
        annual_return = self._annualized_return(returns)
        volatility = returns.std() * np.sqrt(252)
        
        metrics = {
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'calmar_ratio': self._calmar_ratio(returns),
            'omega_ratio': self._omega_ratio(returns),
            'information_ratio': self._information_ratio(returns, benchmark) if benchmark is not None else None
        }
        
        return {k: v for k, v in metrics.items() if v is not None}
    
    def _sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.03) -> float:
        """Sharpe比率"""
        excess_returns = returns - risk_free / 252
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def _sortino_ratio(self, returns: pd.Series, mar: float = 0) -> float:
        """Sortino比率"""
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return 0
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std == 0:
            return 0
        return (returns.mean() - mar) / downside_std * np.sqrt(252)
    
    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calmar比率"""
        annual_return = self._annualized_return(returns)
        max_dd = self._max_drawdown(returns)
        if max_dd == 0:
            return 0
        return annual_return / max_dd
    
    def _omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Omega比率"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf
        
        return gains.sum() / losses.sum()
    
    def _information_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """信息比率"""
        active_returns = returns - benchmark
        if active_returns.std() == 0:
            return 0
        return active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """最大回撤"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def attribution_analysis(self, 
                           portfolio_returns: pd.DataFrame,
                           factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """归因分析"""
        # 简化的Brinson归因分析
        attribution = {}
        
        for factor in factor_returns.columns:
            # 计算因子贡献
            correlation = portfolio_returns.corrwith(factor_returns[factor])
            contribution = correlation * factor_returns[factor].std() / portfolio_returns.std()
            attribution[factor] = float(contribution.mean())
        
        # 计算残差
        explained = sum(attribution.values())
        attribution['residual'] = 1.0 - explained
        
        return attribution


# ==================== 数据管理系统 ====================

class DataManager:
    """数据管理系统 - 数据清洗、存储、更新、质量控制"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('db_path', './data/qilin.db')
        self.cache = {}
        self._init_database()
        
    def _init_database(self):
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP
        ''')
        
        self.conn.commit()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 删除重复
        df = df.drop_duplicates()
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 异常值处理
        df = self._handle_outliers(df)
        
        # 数据类型转换
        df = self._convert_types(df)
        
        # 数据验证
        self._validate_data(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 向前填充价格数据
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # 成交量用0填充
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """处理异常值"""
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            
            # Z-score方法
            z_scores = np.abs((df[col] - mean) / std)
            df = df[z_scores < threshold]
        
        return df
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """数据验证"""
        # 价格合理性检查
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_rows = df[(df['high'] < df['low']) | 
                            (df['high'] < df['open']) | 
                            (df['high'] < df['close']) |
                            (df['low'] > df['open']) | 
                            (df['low'] > df['close'])]
            
            if not invalid_rows.empty:
                logger.warning(f"Found {len(invalid_rows)} rows with invalid OHLC data")
                df = df.drop(invalid_rows.index)
        
        # 成交量检查
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]
        
        return df
    
    def save_data(self, df: pd.DataFrame, table: str = 'market_data'):
        """保存数据"""
        df.to_sql(table, self.conn, if_exists='append', index=False)
        self.conn.commit()
        
        # 更新缓存
        cache_key = f"{table}_{datetime.now().date()}"
        self.cache[cache_key] = df
    
    def load_data(self, 
                 symbol: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """加载数据"""
        query = "SELECT * FROM market_data WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        df = pd.read_sql_query(query, self.conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def update_data(self, source: str = 'tushare'):
        """更新数据"""
        logger.info(f"Updating data from {source}")
        
        # 这里应该实现实际的数据获取逻辑
        # 例如从Tushare或Yahoo Finance获取数据
        
        # 示例：生成模拟数据
        dates = pd.date_range(start=datetime.now() - timedelta(days=1), end=datetime.now(), freq='D')
        
        for date in dates:
            df = pd.DataFrame({
                'symbol': ['000001', '000002', '600000'],
                'date': date,
                'open': np.random.uniform(10, 20, 3),
                'high': np.random.uniform(10, 20, 3),
                'low': np.random.uniform(10, 20, 3),
                'close': np.random.uniform(10, 20, 3),
                'volume': np.random.randint(1000000, 10000000, 3)
            })
            
            # 清洗并保存
            df = self.clean_data(df)
            self.save_data(df)
        
        logger.info("Data update completed")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """数据质量报告"""
        cursor = self.conn.cursor()
        
        # 统计信息
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
        num_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM market_data")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM market_data")
        date_range = cursor.fetchone()
        
        return {
            'num_symbols': num_symbols,
            'total_records': total_records,
            'date_range': {
                'start': date_range[0],
                'end': date_range[1]
            },
            'cache_size': len(self.cache),
            'database_size': Path(self.db_path).stat().st_size / 1024 / 1024  # MB
        }


# ==================== 系统配置与部署 ====================

class SystemDeployment:
    """系统配置与部署 - 配置文件、Docker部署、系统集成"""
    
    def __init__(self):
        self.config_path = Path('./config')
        self.config_path.mkdir(exist_ok=True)
        
    def create_config_files(self):
        """创建配置文件"""
        # 主配置
        main_config = {
            'system': {
                'name': 'Qilin Trading Stack',
                'version': '1.0.0',
                'environment': 'production'
            },
            'database': {
                'type': 'sqlite',
                'path': './data/qilin.db'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'trading': {
                'initial_capital': 1000000,
                'commission_rate': 0.0003,
                'slippage': 0.0001,
                'max_position_size': 0.3
            },
            'risk': {
                'var_confidence': 0.95,
                'max_drawdown': 0.15,
                'position_limit': 0.2
            },
            'data_sources': {
                'primary': 'tushare',
                'backup': 'yahoo_finance',
                'update_frequency': '1D'
            }
        }
        
        # 保存YAML配置
        with open(self.config_path / 'config.yaml', 'w') as f:
            yaml.dump(main_config, f, default_flow_style=False)
        
        # 保存JSON配置
        with open(self.config_path / 'config.json', 'w') as f:
            json.dump(main_config, f, indent=2)
        
        logger.info("Configuration files created")
    
    def create_dockerfile(self):
        """创建Dockerfile"""
        dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/logs /app/config

# Expose ports
EXPOSE 8501 8888 6379

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "main.py"]
'''
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        logger.info("Dockerfile created")
    
    def create_docker_compose(self):
        """创建docker-compose.yml"""
        docker_compose = {
            'version': '3.8',
            'services': {
                'qilin-app': {
                    'build': '.',
                    'container_name': 'qilin-trading',
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs',
                        './config:/app/config'
                    ],
                    'ports': [
                        '8501:8501',
                        '8888:8888'
                    ],
                    'environment': [
                        'REDIS_HOST=redis',
                        'DB_PATH=/app/data/qilin.db'
                    ],
                    'depends_on': [
                        'redis',
                        'postgres'
                    ],
                    'restart': 'unless-stopped'
                },
                'redis': {
                    'image': 'redis:alpine',
                    'container_name': 'qilin-redis',
                    'ports': ['6379:6379'],
                    'volumes': ['redis-data:/data'],
                    'restart': 'unless-stopped'
                },
                'postgres': {
                    'image': 'postgres:13-alpine',
                    'container_name': 'qilin-db',
                    'environment': [
                        'POSTGRES_DB=qilin',
                        'POSTGRES_USER=qilin',
                        'POSTGRES_PASSWORD=qilin123'
                    ],
                    'volumes': ['postgres-data:/var/lib/postgresql/data'],
                    'ports': ['5432:5432'],
                    'restart': 'unless-stopped'
                },
                'web': {
                    'build': '.',
                    'container_name': 'qilin-web',
                    'command': 'streamlit run web/unified_dashboard.py',
                    'ports': ['8501:8501'],
                    'volumes': ['./web:/app/web'],
                    'depends_on': ['qilin-app'],
                    'restart': 'unless-stopped'
                }
            },
            'volumes': {
                'redis-data': {},
                'postgres-data': {}
            },
            'networks': {
                'default': {
                    'name': 'qilin-network'
                }
            }
        }
        
        with open('docker-compose.yml', 'w') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)
        
        logger.info("docker-compose.yml created")
    
    def create_requirements_txt(self):
        """创建requirements.txt"""
        requirements = [
            'pandas==2.0.3',
            'numpy==1.24.3',
            'scipy==1.10.1',
            'scikit-learn==1.3.0',
            'qlib==0.9.0',
            'streamlit==1.25.0',
            'plotly==5.15.0',
            'redis==4.6.0',
            'asyncio==3.4.3',
            'aiohttp==3.8.5',
            'websocket-client==1.6.1',
            'tushare==1.2.89',
            'yfinance==0.2.28',
            'optuna==3.3.0',
            'torch==2.0.1',
            'transformers==4.31.0',
            'langchain==0.0.254',
            'openai==0.27.8',
            'GitPython==3.1.32',
            'docker==6.1.3',
            'pyyaml==6.0.1',
            'python-dotenv==1.0.0',
            'pytest==7.4.0',
            'black==23.7.0',
            'flake8==6.1.0'
        ]
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(requirements))
        
        logger.info("requirements.txt created")
    
    def deploy_system(self, method: str = 'docker'):
        """部署系统"""
        if method == 'docker':
            self._deploy_with_docker()
        elif method == 'kubernetes':
            self._deploy_with_kubernetes()
        else:
            self._deploy_local()
    
    def _deploy_with_docker(self):
        """Docker部署"""
        try:
            client = docker.from_env()
            
            # 构建镜像
            logger.info("Building Docker image...")
            image, logs = client.images.build(
                path='.',
                tag='qilin-trading:latest'
            )
            
            # 运行容器
            logger.info("Starting containers...")
            client.containers.run(
                'qilin-trading:latest',
                detach=True,
                ports={
                    '8501/tcp': 8501,
                    '8888/tcp': 8888
                },
                volumes={
                    f'{Path.cwd()}/data': {'bind': '/app/data', 'mode': 'rw'},
                    f'{Path.cwd()}/logs': {'bind': '/app/logs', 'mode': 'rw'}
                },
                name='qilin-trading',
                restart_policy={'Name': 'unless-stopped'}
            )
            
            logger.info("System deployed successfully with Docker")
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
    
    def _deploy_with_kubernetes(self):
        """Kubernetes部署"""
        # 这里应该实现K8s部署逻辑
        logger.info("Kubernetes deployment not implemented yet")
    
    def _deploy_local(self):
        """本地部署"""
        logger.info("Starting local deployment...")
        
        # 安装依赖
        import subprocess
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
        
        # 初始化数据库
        dm = DataManager({'db_path': './data/qilin.db'})
        
        # 启动Redis（如果可能）
        try:
            subprocess.Popen(['redis-server'])
        except Exception:
            logger.warning("Could not start Redis server")
        
        logger.info("Local deployment completed")
    
    def system_health_check(self) -> Dict[str, bool]:
        """系统健康检查"""
        health = {}
        
        # 检查数据库
        try:
            conn = sqlite3.connect('./data/qilin.db')
            conn.execute("SELECT 1")
            conn.close()
            health['database'] = True
        except Exception:
            health['database'] = False
        
        # 检查Redis
        try:
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            health['redis'] = True
        except Exception:
            health['redis'] = False
        
        # 检查文件系统
        health['filesystem'] = all([
            Path('./data').exists(),
            Path('./logs').exists(),
            Path('./config').exists()
        ])
        
        # 检查网络
        try:
            import requests
            requests.get('https://www.google.com', timeout=5)
            health['network'] = True
        except Exception:
            health['network'] = False
        
        return health


if __name__ == "__main__":
    # 测试代码
    logger.info("Initializing Qilin Trading Stack integrated systems...")
    
    # 创建配置和部署
    deployment = SystemDeployment()
    deployment.create_config_files()
    deployment.create_dockerfile()
    deployment.create_docker_compose()
    deployment.create_requirements_txt()
    
    # 健康检查
    health = deployment.system_health_check()
    logger.info(f"System health: {health}")
    
    # 初始化各系统
    config = {
        'initial_capital': 1000000,
        'var_confidence': 0.95,
        'db_path': './data/qilin.db'
    }
    
    risk_mgmt = RiskManagementSystem(config)
    backtest = BacktestEngine(config)
    perf_analyzer = PerformanceAnalyzer()
    data_mgr = DataManager(config)
    
    logger.info("All systems initialized successfully!")
