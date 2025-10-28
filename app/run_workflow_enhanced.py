#!/usr/bin/env python
"""
麒麟量化系统 - 增强版主工作流
支持并行执行、回测模式、多数据源和详细日志
"""

import asyncio
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
import akshare as ak
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from app.agents.trading_agents_impl import (
    MarketContext,
    IntegratedDecisionAgent,
    analyze_stock,
    batch_analyze
)

# 配置日志
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志系统"""
    log_format = '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    return logging.getLogger("QilinWorkflow")



# v2.1 升级：初始化Qlib，为后续真实数据接入做准备
try:
    # Qlib数据路径 - v2.1已根据项目结构确认
    provider_uri = "G:/test/qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region="cn")
    print(f"Qlib 初始化成功，数据路径: {provider_uri}")
except Exception as e:
    print(f"Qlib 初始化失败，请检查数据路径配置: {e}")
    # 在没有Qlib环境时，系统仍可作为框架运行，但数据加载会失败


class DataManager:
    """数据管理器 - 统一管理多源数据"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("DataManager")
        self.cache = {}
        
    async def load_market_data(
        self,
        symbols: List[str],
        date: Optional[str] = None,
        backtest_mode: bool = False
    ) -> Dict[str, Any]:
        """
        加载市场数据
        
        Args:
            symbols: 股票代码列表
            date: 日期（回测模式用）
            backtest_mode: 是否回测模式
            
        Returns:
            市场数据字典
        """
        start_time = time.time()
        self.logger.info(f"开始加载数据 - 模式: {'回测' if backtest_mode else '实盘'}, 股票数: {len(symbols)}")
        
        # 并行加载各类数据
        tasks = [
            self._load_ohlcv_data(symbols, date, backtest_mode),
            self._load_news_data(symbols, date, backtest_mode),
            self._load_lhb_data(symbols, date, backtest_mode),
            self._load_money_flow_data(symbols, date, backtest_mode),
            self._load_sector_data(symbols, date, backtest_mode),
            self._load_technical_indicators(symbols, date, backtest_mode),
            self._load_fundamental_data(symbols, date, backtest_mode),
            self._load_market_mood(date, backtest_mode)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 整合数据
        market_data = {
            'ohlcv_data': results[0],
            'news_data': results[1],
            'lhb_data': results[2],
            'money_flow_data': results[3],
            'sector_data': results[4],
            'technical_data': results[5],
            'fundamental_data': results[6],
            'market_mood': results[7],
            'timestamp': datetime.now().isoformat()
        }
        
        elapsed = time.time() - start_time
        self.logger.info(f"数据加载完成，耗时: {elapsed:.2f}秒")
        
        return market_data
    
    async def _load_ohlcv_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载OHLCV数据 - v2.1已改造为使用Qlib"""
        self.logger.info(f"使用Qlib加载OHLCV数据，股票数: {len(symbols)}")
        try:
            if not symbols:
                return {}

            # 定义需要获取的特征
            field_names = ['$open', '$high', '$low', '$close', '$volume', '$turnover']
            
            # 确定查询日期
            end_date = pd.to_datetime(date) if date else pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=90) # 默认加载前90天的数据

            # 使用 qlib.data.D.features API 来获取数据
            # freq="day" 表示日线数据
            df = D.features(symbols, field_names, start_time=start_date.strftime('%Y-%m-%d'), end_time=end_date.strftime('%Y-%m-%d'), freq='day')
            
            if df.empty:
                self.logger.warning("Qlib未能加载到任何OHLCV数据")
                return {}

            # Qlib返回的是一个MultiIndex的DataFrame，我们需要按股票代码将其拆分为字典
            data = {symbol: df.xs(symbol, level=1) for symbol in symbols}
            
            self.logger.info(f"成功从Qlib加载了 {len(data)} 只股票的OHLCV数据")
            return data
            
        except Exception as e:
            self.logger.error(f"使用Qlib加载OHLCV数据失败: {e}")
            self.logger.error("请确认：1. Qlib已正确初始化；2. 数据路径正确；3. 数据已下载至对应路径。")
            return {}
    
    async def _load_news_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载新闻数据"""
        try:
            # v2.1新增：使用akshare获取新闻数据示例（默认注释）
            # aiohttp库可能需要安装: pip install aiohttp
            # news_df = ak.stock_news_em(stock="600519") #以单只股票为例
            # for symbol in symbols:
            #     pass #... 遍历并处理
            self.logger.debug("加载新闻数据: (当前为模拟)")
            return {symbol: [] for symbol in symbols}
            
        except Exception as e:
            self.logger.error(f"加载新闻数据失败: {e}")
            return {}
    
    async def _load_lhb_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载龙虎榜数据"""
        try:
            # v2.1新增：使用akshare获取龙虎榜数据示例（默认注释）
            # lhb_df = ak.stock_lhb_detail_em(start_date="20230901", end_date="20230907")
            # for symbol in symbols:
            #     pass #... 遍历并处理
            self.logger.debug("加载龙虎榜数据: (当前为模拟)")
            return {symbol: {'net_buy': 0} for symbol in symbols}
            
        except Exception as e:
            self.logger.error(f"加载龙虎榜数据失败: {e}")
            return {}
    
    async def _load_money_flow_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载资金流向数据"""
        try:
            # v2.1新增：使用akshare获取资金流数据示例（默认注释）
            # for symbol in symbols: 
            #     money_df = ak.stock_individual_fund_flow(stock=symbol, market="sz") # market: sz/sh
            #     pass #... 遍历并处理
            self.logger.debug("加载资金流向数据: (当前为模拟)")
            return {symbol: {} for symbol in symbols}
            
        except Exception as e:
            self.logger.error(f"加载资金流向数据失败: {e}")
            return {}
    
    async def _load_sector_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载板块数据"""
        try:
            # TODO: 实现真实的板块数据加载逻辑
            # 可以通过 Tushare Pro, Akshare, 或其他数据源获取板块涨幅、热度、排名等信息
            # 此处返回空数据作为示例
            self.logger.debug("加载板块数据: (当前为模拟)")
            return {}
            
        except Exception as e:
            self.logger.error(f"加载板块数据失败: {e}")
            return {}
    
    async def _load_technical_indicators(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载技术指标"""
        try:
            # TODO: 实现真实的技术指标计算/加载逻辑
            # 大部分技术指标 (如RSI, MACD) 可以基于Qlib加载的OHLCV数据进行计算
            # 另一部分 (如封单比, 竞价波动) 需要更高频的数据源
            # 此处返回空数据作为示例
            self.logger.debug("加载技术指标: (当前为模拟)")
            return {symbol: {} for symbol in symbols}

        except Exception as e:
            self.logger.error(f"加载技术指标失败: {e}")
            return {}
    
    async def _load_fundamental_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载基本面数据"""
        try:
            # TODO: 实现真实的基本面数据加载逻辑
            # 可以通过 Qlib 的特征库或 Tushare Pro 的接口获取
            # 此处返回空数据作为示例
            self.logger.debug("加载基本面数据: (当前为模拟)")
            return {symbol: {} for symbol in symbols}
            
        except Exception as e:
            self.logger.error(f"加载基本面数据失败: {e}")
            return {}
    
    async def _load_market_mood(self, date: Optional[str], backtest: bool) -> float:
        """加载市场情绪指数"""
        try:
            # TODO: 实现真实的市场情绪计算逻辑
            # 可以通过计算全市场涨跌比、成交量、炸板率等指标综合得出
            # 此处返回一个中性值50.0
            self.logger.debug("加载市场情绪: (当前为模拟)")
            return 50.0
        except Exception as e:
            self.logger.error(f"加载市场情绪失败: {e}")
            return 50.0


class TradingWorkflow:
    """主交易工作流"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("TradingWorkflow")
        self.data_manager = DataManager(config)
        self.decision_agent = IntegratedDecisionAgent()
        self.execution_history = []
        self.operational_command = {} # v2.1 新增：用于存储元帅的作战指令

    def set_operational_command(self, command: Dict[str, Any]):
        """ v2.1 升级：接收并应用元帅的作战指令 """
        self.logger.info(f"接收到新的作战指令: {command.get('market_regime', '未知')}")
        self.operational_command = command
        # TODO: 未来可将指令传递给决策Agent，以动态调整其内部参数
        # self.decision_agent.set_operational_command(command)
        
    async def run(
        self,
        symbols: List[str],
        mode: str = "live",
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行交易工作流
        
        Args:
            symbols: 股票代码列表
            mode: 运行模式 (live/backtest)
            date: 回测日期
            
        Returns:
            执行结果
        """
        start_time = time.time()
        self.logger.info(f"{'='*80}")
        self.logger.info(f"启动交易工作流 - 模式: {mode}, 股票数: {len(symbols)}")
        self.logger.info(f"时间: {datetime.now()}")
        self.logger.info(f"{'='*80}")
        
        try:
            # 1. 加载市场数据
            self.logger.info("步骤1: 加载市场数据...")
            market_data = await self.data_manager.load_market_data(
                symbols=symbols,
                date=date,
                backtest_mode=(mode == "backtest")
            )
            
            # 2. 准备分析上下文
            self.logger.info("步骤2: 准备分析上下文...")
            contexts = self._prepare_contexts(symbols, market_data)
            
            # 3. 并行分析所有股票
            self.logger.info(f"步骤3: 并行分析{len(symbols)}只股票...")
            analysis_start = time.time()
            
            # 创建分析任务
            tasks = []
            for symbol in symbols:
                ctx = contexts[symbol]
                task = self.decision_agent.analyze_parallel(symbol, ctx)
                tasks.append(task)
            
            # 并行执行
            results = await asyncio.gather(*tasks)
            
            analysis_time = time.time() - analysis_start
            self.logger.info(f"分析完成，总耗时: {analysis_time:.2f}秒，平均每只: {analysis_time/len(symbols):.2f}秒")
            
            # 4. 生成交易信号
            self.logger.info("步骤4: 生成交易信号...")
            signals = self._generate_signals(results)
            
            # 5. 执行交易（实盘或模拟）
            if mode == "live":
                self.logger.info("步骤5: 执行实盘交易...")
                execution_results = await self._execute_trades(signals)
            else:
                self.logger.info("步骤5: 模拟交易执行...")
                execution_results = self._simulate_trades(signals)
            
            # 6. 记录和报告
            total_time = time.time() - start_time
            report = self._generate_report(results, signals, execution_results, total_time)
            
            self.logger.info(f"{'='*80}")
            self.logger.info(f"工作流执行完成，总耗时: {total_time:.2f}秒")
            self.logger.info(f"{'='*80}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"工作流执行失败: {e}")
            raise
    
    def _prepare_contexts(self, symbols: List[str], market_data: Dict) -> Dict[str, MarketContext]:
        """准备每个股票的分析上下文"""
        contexts = {}
        
        for symbol in symbols:
            # 获取各类数据
            ohlcv = market_data['ohlcv_data'].get(symbol, pd.DataFrame())
            news = market_data['news_data'].get(symbol, [])
            lhb = market_data['lhb_data'].get(symbol, {})
            money_flow = market_data['money_flow_data'].get(symbol, {})
            tech = market_data['technical_data'].get(symbol, {})
            fundamental = market_data['fundamental_data'].get(symbol, {})
            
            # 创建上下文
            ctx = MarketContext(
                ohlcv=ohlcv,
                news_titles=news,
                lhb_netbuy=lhb.get('net_buy', 0),
                market_mood_score=market_data['market_mood'],
                sector_heat=market_data['sector_data'],
                money_flow=money_flow,
                technical_indicators=tech,
                fundamental_data=fundamental
            )
            
            contexts[symbol] = ctx
            
        self.logger.debug(f"准备了{len(contexts)}个股票的分析上下文")
        return contexts
    
    def _generate_signals(self, results: List[Dict]) -> List[Dict]:
        """生成交易信号"""
        signals = []
        
        # 按得分排序
        sorted_results = sorted(results, key=lambda x: x['weighted_score'], reverse=True)
        
        # 选择前N只
        max_positions = self.config.get('max_positions', 5)
        
        for i, result in enumerate(sorted_results[:max_positions]):
            decision = result['decision']
            
            if decision['action'] in ['strong_buy', 'buy']:
                signal = {
                    'rank': i + 1,
                    'symbol': result['symbol'],
                    'action': decision['action'],
                    'score': result['weighted_score'],
                    'position': decision['position'],
                    'confidence': decision['confidence'],
                    'reason': decision['reason'],
                    'risk_level': decision['risk_level'],
                    'timestamp': datetime.now().isoformat()
                }
                signals.append(signal)
                
                self.logger.info(
                    f"信号{i+1}: {signal['symbol']} - {signal['action']} "
                    f"(得分:{signal['score']:.2f}, 仓位:{signal['position']}, 风险:{signal['risk_level']})"
                )
        
        return signals
    
    async def _execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """执行实盘交易"""
        execution_results = []
        
        for signal in signals:
            try:
                # 这里应该调用实际的交易API
                self.logger.info(f"执行交易: {signal['symbol']} - {signal['action']}")
                
                result = {
                    'symbol': signal['symbol'],
                    'action': signal['action'],
                    'status': 'success',
                    'order_id': f"ORD_{signal['symbol']}_{int(time.time())}",
                    'executed_price': None,  # 实际成交价
                    'executed_volume': None,  # 实际成交量
                    'timestamp': datetime.now().isoformat()
                }
                execution_results.append(result)
                
            except Exception as e:
                self.logger.error(f"交易执行失败 {signal['symbol']}: {e}")
                result = {
                    'symbol': signal['symbol'],
                    'action': signal['action'],
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                execution_results.append(result)
        
        return execution_results
    
    def _simulate_trades(self, signals: List[Dict]) -> List[Dict]:
        """模拟交易执行"""
        execution_results = []
        
        for signal in signals:
            self.logger.info(f"模拟交易: {signal['symbol']} - {signal['action']}")
            
            result = {
                'symbol': signal['symbol'],
                'action': signal['action'],
                'status': 'simulated',
                'simulated_price': np.random.uniform(10, 11),
                'simulated_volume': 1000,
                'timestamp': datetime.now().isoformat()
            }
            execution_results.append(result)
        
        return execution_results
    
    def _generate_report(
        self,
        analysis_results: List[Dict],
        signals: List[Dict],
        execution_results: List[Dict],
        total_time: float
    ) -> Dict[str, Any]:
        """生成执行报告"""
        
        # 统计信息
        total_analyzed = len(analysis_results)
        total_signals = len(signals)
        total_executed = len([r for r in execution_results if r.get('status') in ['success', 'simulated']])
        
        # 得分统计
        scores = [r['weighted_score'] for r in analysis_results]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        report = {
            'summary': {
                'total_analyzed': total_analyzed,
                'total_signals': total_signals,
                'total_executed': total_executed,
                'execution_time': total_time,
                'avg_analysis_time': total_time / total_analyzed if total_analyzed > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'scores': {
                'average': avg_score,
                'max': max_score,
                'min': min_score,
                'distribution': {
                    'excellent': len([s for s in scores if s >= 80]),
                    'good': len([s for s in scores if 60 <= s < 80]),
                    'moderate': len([s for s in scores if 40 <= s < 60]),
                    'poor': len([s for s in scores if s < 40])
                }
            },
            'top_stocks': [
                {
                    'rank': i + 1,
                    'symbol': r['symbol'],
                    'score': r['weighted_score'],
                    'decision': r['decision']['action']
                }
                for i, r in enumerate(sorted(analysis_results, key=lambda x: x['weighted_score'], reverse=True)[:10])
            ],
            'signals': signals,
            'execution': execution_results,
            'details': {
                'analysis_results': analysis_results,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 打印报告摘要
        self.logger.info("\n" + "="*60)
        self.logger.info("执行报告摘要")
        self.logger.info("="*60)
        self.logger.info(f"分析股票数: {total_analyzed}")
        self.logger.info(f"生成信号数: {total_signals}")
        self.logger.info(f"执行交易数: {total_executed}")
        self.logger.info(f"平均得分: {avg_score:.2f}")
        self.logger.info(f"最高得分: {max_score:.2f}")
        self.logger.info(f"总耗时: {total_time:.2f}秒")
        self.logger.info("="*60)
        
        return report


async def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="麒麟量化系统 - 增强版工作流")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live", help="运行模式")
    parser.add_argument("--date", type=str, help="回测日期 (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="+", help="股票代码列表")
    parser.add_argument("--config", type=str, default="config/workflow.json", help="配置文件路径")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--max-positions", type=int, default=5, help="最大持仓数")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level, args.log_file)
    
    # 加载配置
    config = {
        'max_positions': args.max_positions,
        'data_sources': {
            'ohlcv': 'tushare',
            'news': 'eastmoney',
            'lhb': 'eastmoney',
            'money_flow': 'sina'
        }
    }
    
    # 如果有配置文件，加载它
    if Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # 股票列表
    # 格式化股票代码，以适配akshare等库 (e.g., 600519 -> sh600519)
    def format_symbol(s):
        if s.startswith('6'):
            return f"sh{s}"
        elif s.startswith(('0', '3')):
            return f"sz{s}"
        return s

    if args.symbols:
        symbols = [format_symbol(s) for s in args.symbols]
    else:
        # 默认股票池
        default_symbols = [
            '600519', '000001', '000858', '300750', '002415',
            '601318', '600036', '000651', '000333', '002714'
        ]
        symbols = [format_symbol(s) for s in default_symbols]
    
    logger.info(f"配置: {config}")
    logger.info(f"股票池 (格式化后): {symbols}")
    
    # 创建并运行工作流
    workflow = TradingWorkflow(config)
    
    try:
        report = await workflow.run(
            symbols=symbols,
            mode=args.mode,
            date=args.date
        )
        
        # 保存报告
        report_file = f"reports/workflow_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        Path("reports").mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"报告已保存至: {report_file}")
        
        return report
        
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        raise


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())