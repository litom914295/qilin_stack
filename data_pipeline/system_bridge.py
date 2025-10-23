"""
三系统数据集成桥接层
为Qlib、TradingAgents、RD-Agent提供统一数据接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .unified_data import (
    UnifiedDataPipeline,
    get_unified_pipeline,
    DataFrequency,
    DataSource
)

logger = logging.getLogger(__name__)


# ============================================================================
# Qlib数据桥接
# ============================================================================

class QlibDataBridge:
    """Qlib数据桥接器"""
    
    def __init__(self, pipeline: Optional[UnifiedDataPipeline] = None):
        self.pipeline = pipeline or get_unified_pipeline()
    
    def get_qlib_format_data(self,
                            instruments: List[str],
                            fields: List[str],
                            start_time: str,
                            end_time: str,
                            freq: str = 'day') -> pd.DataFrame:
        """
        获取Qlib格式数据
        
        Args:
            instruments: 股票列表
            fields: 字段列表（如$open, $close等）
            start_time: 开始时间
            end_time: 结束时间
            freq: 频率
        
        Returns:
            MultiIndex DataFrame (instrument, datetime)
        """
        # 获取原始数据
        data = self.pipeline.get_bars(
            symbols=instruments,
            start_date=start_time,
            end_date=end_time,
            frequency=DataFrequency.DAY
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # 转换字段名
        qlib_columns = {}
        for field in fields:
            clean_field = field.replace('$', '')
            if clean_field in data.columns:
                qlib_columns[clean_field] = field
        
        # 重命名并选择列
        if qlib_columns:
            data = data.rename(columns={v: k for k, v in qlib_columns.items()})
            data = data[[col for col in fields if col in data.columns]]
        
        return data
    
    def get_features_for_model(self,
                               instruments: List[str],
                               start_time: str,
                               end_time: str) -> pd.DataFrame:
        """
        获取模型训练特征
        
        Returns:
            完整特征DataFrame，包含技术指标
        """
        # 获取基础数据
        data = self.pipeline.get_bars(
            symbols=instruments,
            start_date=start_time,
            end_date=end_time,
            frequency=DataFrequency.DAY
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # 添加技术指标
        data = self._add_technical_indicators(data)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        # 确保数据按symbol和date排序
        data = data.sort_index()
        
        # 收益率
        data['returns'] = data.groupby(level=0)['close'].pct_change()
        
        # 5日均线
        data['ma5'] = data.groupby(level=0)['close'].rolling(5).mean().values
        
        # 20日均线
        data['ma20'] = data.groupby(level=0)['close'].rolling(20).mean().values
        
        # 波动率
        data['volatility'] = data.groupby(level=0)['returns'].rolling(20).std().values
        
        # RSI
        data['rsi'] = data.groupby(level=0).apply(
            lambda x: self._calculate_rsi(x['close'], 14)
        ).values
        
        return data
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# ============================================================================
# TradingAgents数据桥接
# ============================================================================

class TradingAgentsDataBridge:
    """TradingAgents数据桥接器"""
    
    def __init__(self, pipeline: Optional[UnifiedDataPipeline] = None):
        self.pipeline = pipeline or get_unified_pipeline()
    
    def get_market_state(self,
                        symbols: List[str],
                        date: str) -> Dict[str, Any]:
        """
        获取市场状态（TradingAgents格式）
        
        Returns:
            {
                'prices': {...},
                'volumes': {...},
                'fundamentals': {...}
            }
        """
        # 获取历史数据（最近30天）
        end_date = pd.to_datetime(date)
        start_date = (end_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        bars = self.pipeline.get_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=date,
            frequency=DataFrequency.DAY
        )
        
        # 获取基本面
        fundamentals = self.pipeline.get_fundamentals(
            symbols=symbols,
            date=date
        )
        
        # 组装市场状态
        market_state = {
            'timestamp': date,
            'prices': {},
            'volumes': {},
            'fundamentals': {}
        }
        
        if not bars.empty:
            for symbol in symbols:
                if symbol in bars.index.get_level_values(0):
                    symbol_data = bars.loc[symbol]
                    
                    market_state['prices'][symbol] = {
                        'current': float(symbol_data['close'].iloc[-1]) if len(symbol_data) > 0 else 0.0,
                        'open': float(symbol_data['open'].iloc[-1]) if len(symbol_data) > 0 else 0.0,
                        'high': float(symbol_data['high'].iloc[-1]) if len(symbol_data) > 0 else 0.0,
                        'low': float(symbol_data['low'].iloc[-1]) if len(symbol_data) > 0 else 0.0,
                        'history': symbol_data['close'].tolist()
                    }
                    
                    market_state['volumes'][symbol] = {
                        'current': float(symbol_data['volume'].iloc[-1]) if len(symbol_data) > 0 else 0.0,
                        'average': float(symbol_data['volume'].mean())
                    }
        
        return market_state
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        获取实时数据
        
        Returns:
            {symbol: {'price': ..., 'volume': ..., 'change': ...}}
        """
        realtime = self.pipeline.get_realtime_quote(symbols)
        
        result = {}
        if not realtime.empty:
            for symbol in symbols:
                # 根据实际数据源格式解析
                result[symbol] = {
                    'price': 0.0,
                    'volume': 0.0,
                    'change': 0.0,
                    'turnover_rate': 0.0
                }
        
        return result


# ============================================================================
# RD-Agent数据桥接
# ============================================================================

class RDAgentDataBridge:
    """RD-Agent数据桥接器"""
    
    def __init__(self, pipeline: Optional[UnifiedDataPipeline] = None):
        self.pipeline = pipeline or get_unified_pipeline()
    
    def get_factor_data(self,
                       symbols: List[str],
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        获取因子数据（RD-Agent格式）
        
        Returns:
            MultiIndex DataFrame with datetime and features
        """
        # 获取原始数据
        bars = self.pipeline.get_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAY
        )
        
        if bars.empty:
            return pd.DataFrame()
        
        # 构造因子
        factor_data = self._construct_factors(bars)
        
        return factor_data
    
    def _construct_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """构造因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 价格因子
        factors['price_to_ma5'] = data['close'] / data.groupby(level=0)['close'].rolling(5).mean().values
        factors['price_to_ma20'] = data['close'] / data.groupby(level=0)['close'].rolling(20).mean().values
        
        # 动量因子
        factors['momentum_5d'] = data.groupby(level=0)['close'].pct_change(5)
        factors['momentum_20d'] = data.groupby(level=0)['close'].pct_change(20)
        
        # 波动率因子
        returns = data.groupby(level=0)['close'].pct_change()
        factors['volatility_20d'] = returns.groupby(level=0).rolling(20).std().values
        
        # 成交量因子
        factors['volume_ratio'] = data['volume'] / data.groupby(level=0)['volume'].rolling(20).mean().values
        
        # 振幅因子
        factors['amplitude'] = (data['high'] - data['low']) / data['low']
        
        return factors
    
    def get_limit_up_data(self,
                         symbols: List[str],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        获取涨停板数据（专用）
        
        Returns:
            涨停板记录DataFrame
        """
        # 获取原始数据
        bars = self.pipeline.get_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAY
        )
        
        if bars.empty:
            return pd.DataFrame()
        
        # 计算涨跌幅
        bars['pct_change'] = bars.groupby(level=0)['close'].pct_change()
        
        # 识别涨停（涨幅>9.5%）
        limit_ups = bars[bars['pct_change'] > 0.095].copy()
        
        # 添加涨停特征
        limit_ups['is_limit_up'] = True
        limit_ups['limit_up_strength'] = (limit_ups['pct_change'] / 0.10) * 100
        
        return limit_ups


# ============================================================================
# 统一桥接管理器
# ============================================================================

class UnifiedDataBridge:
    """统一数据桥接管理器"""
    
    def __init__(self):
        self.pipeline = get_unified_pipeline()
        
        # 初始化各系统桥接器
        self.qlib_bridge = QlibDataBridge(self.pipeline)
        self.ta_bridge = TradingAgentsDataBridge(self.pipeline)
        self.rd_bridge = RDAgentDataBridge(self.pipeline)
        
        logger.info("✅ 统一数据桥接管理器初始化完成")
    
    def get_qlib_bridge(self) -> QlibDataBridge:
        """获取Qlib桥接器"""
        return self.qlib_bridge
    
    def get_tradingagents_bridge(self) -> TradingAgentsDataBridge:
        """获取TradingAgents桥接器"""
        return self.ta_bridge
    
    def get_rdagent_bridge(self) -> RDAgentDataBridge:
        """获取RD-Agent桥接器"""
        return self.rd_bridge
    
    def get_unified_pipeline(self) -> UnifiedDataPipeline:
        """获取底层统一管道"""
        return self.pipeline
    
    def test_all_bridges(self) -> Dict[str, bool]:
        """测试所有桥接器"""
        results = {}
        
        test_symbols = ['000001.SZ']
        test_date = '2024-01-10'
        
        # 测试Qlib桥接
        try:
            qlib_data = self.qlib_bridge.get_qlib_format_data(
                instruments=test_symbols,
                fields=['$close', '$volume'],
                start_time='2024-01-01',
                end_time=test_date,
                freq='day'
            )
            results['qlib_bridge'] = not qlib_data.empty
        except Exception as e:
            logger.error(f"Qlib桥接测试失败: {e}")
            results['qlib_bridge'] = False
        
        # 测试TradingAgents桥接
        try:
            ta_state = self.ta_bridge.get_market_state(test_symbols, test_date)
            results['tradingagents_bridge'] = 'prices' in ta_state
        except Exception as e:
            logger.error(f"TradingAgents桥接测试失败: {e}")
            results['tradingagents_bridge'] = False
        
        # 测试RD-Agent桥接
        try:
            rd_factors = self.rd_bridge.get_factor_data(
                symbols=test_symbols,
                start_date='2024-01-01',
                end_date=test_date
            )
            results['rdagent_bridge'] = not rd_factors.empty
        except Exception as e:
            logger.error(f"RD-Agent桥接测试失败: {e}")
            results['rdagent_bridge'] = False
        
        return results


# ============================================================================
# 工厂函数
# ============================================================================

_bridge_instance = None

def get_unified_bridge() -> UnifiedDataBridge:
    """获取统一数据桥接管理器单例"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = UnifiedDataBridge()
    return _bridge_instance


# ============================================================================
# 测试
# ============================================================================

import logging
logger = logging.getLogger(__name__)

def test_data_bridge():
    """测试数据桥接"""
    logger.info("三系统数据桥接测试")
    
    bridge = get_unified_bridge()
    
    # 测试所有桥接器
    logger.info("1️⃣ 测试桥接器连通性:")
    results = bridge.test_all_bridges()
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {name}: {'正常' if status else '失败'}")
    
    # 详细测试Qlib桥接
    logger.info("2️⃣ 测试Qlib桥接:")
    qlib_bridge = bridge.get_qlib_bridge()
    qlib_data = qlib_bridge.get_features_for_model(
        instruments=['000001.SZ'],
        start_time='2024-01-01',
        end_time='2024-01-10'
    )
    
    if not qlib_data.empty:
        logger.info(f"  获取到 {len(qlib_data)} 条数据")
        logger.info(f"  特征数量: {len(qlib_data.columns)}")
        logger.info(f"  特征列表: {list(qlib_data.columns[:5])}")
    else:
        logger.info("  ❌ 未获取到数据")
    
    # 详细测试TradingAgents桥接
    logger.info("3️⃣ 测试TradingAgents桥接:")
    ta_bridge = bridge.get_tradingagents_bridge()
    market_state = ta_bridge.get_market_state(['000001.SZ'], '2024-01-10')
    
    logger.info(f"  市场状态包含: {list(market_state.keys())}")
    if 'prices' in market_state and market_state['prices']:
        logger.info(f"  价格数据: {list(market_state['prices'].keys())}")
    
    # 详细测试RD-Agent桥接
    logger.info("4️⃣ 测试RD-Agent桥接:")
    rd_bridge = bridge.get_rdagent_bridge()
    factors = rd_bridge.get_factor_data(
        symbols=['000001.SZ'],
        start_date='2024-01-01',
        end_date='2024-01-10'
    )
    
    if not factors.empty:
        logger.info(f"  获取到 {len(factors)} 条因子数据")
        logger.info(f"  因子数量: {len(factors.columns)}")
        logger.info(f"  因子列表: {list(factors.columns)}")
    else:
        logger.info("  ❌ 未获取到数据")
    
    logger.info("✅ 测试完成")


if __name__ == "__main__":
    from app.core.logging_setup import setup_logging
    setup_logging()
    test_data_bridge()
