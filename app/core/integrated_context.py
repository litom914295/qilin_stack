"""
麒麟量化系统 - 集成数据上下文管理器
将TradingContext与RealTimeDataFetcher集成
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

from app.core.trading_context import TradingContext
from app.data.realtime_data_fetcher import RealTimeDataFetcher

logger = logging.getLogger(__name__)


class IntegratedTradingContext:
    """集成的交易上下文管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化集成上下文
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.context = TradingContext()
        self.data_fetcher = RealTimeDataFetcher(config)
        
        # 当前交易日期和股票
        self.current_date = None
        self.current_symbol = None
        
        # 数据缓存
        self.data_cache = {}
        
    async def initialize(
        self,
        symbol: str,
        trading_date: Optional[datetime] = None,
        lookback_days: int = 30
    ):
        """
        初始化交易环境
        
        Args:
            symbol: 股票代码
            trading_date: 交易日期
            lookback_days: 历史数据天数
        """
        self.current_symbol = symbol
        self.current_date = trading_date or datetime.now()
        
        # 计算日期范围
        end_date = self.current_date
        start_date = end_date - timedelta(days=lookback_days * 1.5)  # 考虑非交易日
        
        # 获取历史数据
        logger.info(f"获取{symbol}历史数据...")
        historical_data = await self.data_fetcher.get_historical_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        
        # 设置到context
        self.context.set_historical_data(historical_data.tail(lookback_days))
        
        # 获取实时数据（如果是当天）
        if self._is_today(trading_date):
            await self.update_realtime_data([symbol])
        
        # 获取市场数据
        await self.update_market_data()
        
        logger.info(f"交易环境初始化完成: {symbol} @ {self.current_date}")
    
    async def update_realtime_data(self, symbols: Optional[List[str]] = None):
        """
        更新实时数据
        
        Args:
            symbols: 股票代码列表，None则使用当前股票
        """
        if symbols is None:
            symbols = [self.current_symbol] if self.current_symbol else []
        
        if not symbols:
            logger.warning("没有指定要更新的股票")
            return
        
        # 获取实时行情
        quotes = await self.data_fetcher.get_realtime_quotes(symbols)
        
        # 更新到context
        for symbol, quote_df in quotes.items():
            if not quote_df.empty:
                self.context.set_realtime_data(quote_df.iloc[0].to_dict())
                logger.debug(f"更新{symbol}实时数据")
    
    async def update_market_data(self):
        """更新市场整体数据"""
        market_data = await self.data_fetcher.get_market_data()
        
        # 转换为context格式
        market_context = {
            'indices': {},
            'market_stats': market_data.get('stats', {}),
            'sectors': market_data.get('sectors', []),
            'lhb': market_data.get('lhb', pd.DataFrame())
        }
        
        # 处理指数数据
        for index_code, index_df in market_data.get('indices', {}).items():
            if not index_df.empty:
                market_context['indices'][index_code] = {
                    'close': index_df.iloc[0]['close'],
                    'pct_change': index_df.iloc[0]['pct_change']
                }
        
        # 更新到context
        self.context.market_data = market_context
        logger.debug("市场数据已更新")
    
    async def get_multi_symbol_data(
        self,
        symbols: List[str],
        lookback_days: int = 30
    ) -> Dict[str, TradingContext]:
        """
        获取多只股票的交易上下文
        
        Args:
            symbols: 股票代码列表
            lookback_days: 历史数据天数
            
        Returns:
            股票上下文字典
        """
        contexts = {}
        
        # 并发获取多只股票数据
        tasks = []
        for symbol in symbols:
            task = self._create_symbol_context(symbol, lookback_days)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"获取{symbol}数据失败: {result}")
                contexts[symbol] = TradingContext()  # 空context
            else:
                contexts[symbol] = result
        
        return contexts
    
    async def _create_symbol_context(
        self,
        symbol: str,
        lookback_days: int
    ) -> TradingContext:
        """为单个股票创建上下文"""
        ctx = TradingContext()
        
        # 获取历史数据
        end_date = self.current_date or datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)
        
        historical_data = await self.data_fetcher.get_historical_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        
        ctx.set_historical_data(historical_data.tail(lookback_days))
        
        # 获取实时数据
        if self._is_today(end_date):
            quotes = await self.data_fetcher.get_realtime_quotes([symbol])
            if symbol in quotes and not quotes[symbol].empty:
                ctx.set_realtime_data(quotes[symbol].iloc[0].to_dict())
        
        return ctx
    
    async def get_sector_stocks(
        self,
        sector: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取板块成分股
        
        Args:
            sector: 板块名称
            limit: 数量限制
            
        Returns:
            股票列表
        """
        # TODO: 实现板块成分股获取
        # 这里需要根据具体数据源实现
        logger.warning(f"板块成分股获取功能待实现: {sector}")
        
        # 返回模拟数据
        return [
            {'symbol': '000001', 'name': '平安银行', 'weight': 0.1},
            {'symbol': '000002', 'name': '万科A', 'weight': 0.08}
        ][:limit]
    
    async def get_similar_stocks(
        self,
        symbol: str,
        limit: int = 5
    ) -> List[str]:
        """
        获取相似股票
        
        Args:
            symbol: 股票代码
            limit: 数量限制
            
        Returns:
            相似股票代码列表
        """
        # TODO: 实现相似股票获取
        # 可以基于行业、市值、技术指标等
        logger.warning(f"相似股票获取功能待实现: {symbol}")
        
        # 返回模拟数据
        return ['000002', '000858', '002142'][:limit]
    
    def get_context(self) -> TradingContext:
        """获取当前交易上下文"""
        return self.context
    
    def _is_today(self, date: Optional[datetime]) -> bool:
        """判断是否为今天"""
        if date is None:
            return True
        return date.date() == datetime.now().date()
    
    async def analyze_with_agents(
        self,
        agents: List[Any],
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用Agent分析
        
        Args:
            agents: Agent列表
            symbol: 股票代码
            
        Returns:
            分析结果
        """
        if symbol:
            await self.initialize(symbol)
        
        results = {}
        
        for agent in agents:
            try:
                # 调用Agent的analyze方法
                if hasattr(agent, 'analyze'):
                    score = agent.analyze(self.context)
                    results[agent.__class__.__name__] = {
                        'score': score,
                        'timestamp': datetime.now()
                    }
                    logger.info(f"{agent.__class__.__name__}分析完成: {score}")
            except Exception as e:
                logger.error(f"{agent.__class__.__name__}分析失败: {e}")
                results[agent.__class__.__name__] = {
                    'score': 0,
                    'error': str(e)
                }
        
        return results
    
    async def batch_analyze(
        self,
        symbols: List[str],
        agents: List[Any],
        lookback_days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量分析多只股票
        
        Args:
            symbols: 股票代码列表
            agents: Agent列表
            lookback_days: 历史数据天数
            
        Returns:
            分析结果字典
        """
        # 获取所有股票的上下文
        contexts = await self.get_multi_symbol_data(symbols, lookback_days)
        
        # 分析每只股票
        all_results = {}
        
        for symbol, context in contexts.items():
            results = {}
            
            for agent in agents:
                try:
                    if hasattr(agent, 'analyze'):
                        score = agent.analyze(context)
                        results[agent.__class__.__name__] = score
                except Exception as e:
                    logger.error(f"{symbol} {agent.__class__.__name__}分析失败: {e}")
                    results[agent.__class__.__name__] = 0
            
            all_results[symbol] = results
            logger.info(f"{symbol}批量分析完成")
        
        return all_results


class DataStreamManager:
    """数据流管理器"""
    
    def __init__(self, integrated_context: IntegratedTradingContext):
        """
        初始化数据流管理器
        
        Args:
            integrated_context: 集成上下文
        """
        self.context = integrated_context
        self.is_streaming = False
        self.stream_task = None
        
    async def start_streaming(
        self,
        symbols: List[str],
        interval: int = 5,
        callback: Optional[callable] = None
    ):
        """
        开始数据流
        
        Args:
            symbols: 股票代码列表
            interval: 更新间隔(秒)
            callback: 数据更新回调函数
        """
        if self.is_streaming:
            logger.warning("数据流已在运行")
            return
        
        self.is_streaming = True
        self.stream_task = asyncio.create_task(
            self._stream_loop(symbols, interval, callback)
        logger.info(f"数据流已启动: {symbols}")
    
    async def _stream_loop(
        self,
        symbols: List[str],
        interval: int,
        callback: Optional[callable]
    ):
        """数据流循环"""
        while self.is_streaming:
            try:
                # 更新实时数据
                await self.context.update_realtime_data(symbols)
                
                # 每5分钟更新一次市场数据
                if asyncio.get_event_loop().time() % 300 < interval:
                    await self.context.update_market_data()
                
                # 调用回调
                if callback:
                    await callback(self.context.get_context())
                
            except Exception as e:
                logger.error(f"数据流更新失败: {e}")
            
            await asyncio.sleep(interval)
    
    async def stop_streaming(self):
        """停止数据流"""
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
            
        logger.info("数据流已停止")


# 使用示例
async def example_usage():
    """使用示例"""
    # 创建集成上下文
    config = {
        'redis_url': 'redis://localhost:6379',
        'cache_ttl': 300,
        'tushare_token': 'your_token_here'
    }
    
    context = IntegratedTradingContext(config)
    
    # 初始化单只股票
    await context.initialize('000001', lookback_days=30)
    
    # 获取上下文
    trading_ctx = context.get_context()
    print(f"历史数据: {len(trading_ctx.historical_data)}条")
    print(f"实时数据: {trading_ctx.realtime_data}")
    
    # 批量分析
    from app.agents.enhanced_agents import (
        EnhancedAuctionGameAgent,
        EnhancedMarketEcologyAgent
    
    agents = [
        EnhancedAuctionGameAgent(),
        EnhancedMarketEcologyAgent()
    ]
    
    results = await context.batch_analyze(
        ['000001', '000002', '000858'],
        agents
    
    print("\n批量分析结果:")
    for symbol, scores in results.items():
        print(f"{symbol}: {scores}")
    
    # 启动数据流
    stream_manager = DataStreamManager(context)
    
    async def on_data_update(ctx):
        print(f"数据更新: {ctx.realtime_data.get('close')}")
    
    await stream_manager.start_streaming(
        ['000001'],
        interval=5,
        callback=on_data_update
    
    # 运行10秒后停止
    await asyncio.sleep(10)
    await stream_manager.stop_streaming()


if __name__ == "__main__":
    asyncio.run(example_usage())