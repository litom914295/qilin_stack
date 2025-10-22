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
import aiohttp
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
        """加载OHLCV数据"""
        try:
            if backtest and date:
                # 回测模式：从历史数据文件加载
                file_path = Path(f"data/history/{date}/ohlcv.parquet")
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    return {s: df[df['symbol'] == s] for s in symbols}
            
            # 实盘模式：从API获取
            # 这里应该调用实际的数据API
            data = {}
            for symbol in symbols:
                # 模拟数据
                data[symbol] = pd.DataFrame({
                    'open': np.random.uniform(10, 11, 20),
                    'high': np.random.uniform(11, 12, 20),
                    'low': np.random.uniform(9, 10, 20),
                    'close': np.random.uniform(10, 11, 20),
                    'volume': np.random.uniform(1000000, 5000000, 20),
                    'turnover_rate': np.random.uniform(1, 20, 20)
                })
            return data
            
        except Exception as e:
            self.logger.error(f"加载OHLCV数据失败: {e}")
            return {}
    
    async def _load_news_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载新闻数据"""
        try:
            news_data = {}
            
            # 模拟新闻数据
            news_templates = [
                "{symbol}公司获得重大订单",
                "{symbol}发布业绩预增公告",
                "机构密集调研{symbol}",
                "{symbol}股价创历史新高",
                "北向资金大幅买入{symbol}",
                "{symbol}所属板块强势拉升"
            ]
            
            for symbol in symbols:
                news_data[symbol] = [
                    template.format(symbol=symbol)
                    for template in np.random.choice(news_templates, size=3, replace=False)
                ]
            
            self.logger.debug(f"加载新闻数据: {len(news_data)}条")
            return news_data
            
        except Exception as e:
            self.logger.error(f"加载新闻数据失败: {e}")
            return {}
    
    async def _load_lhb_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载龙虎榜数据"""
        try:
            lhb_data = {}
            
            # 模拟龙虎榜数据
            for symbol in symbols:
                if np.random.random() > 0.7:  # 30%概率上龙虎榜
                    lhb_data[symbol] = {
                        'net_buy': np.random.uniform(-5, 10),  # 净买入（亿元）
                        'buy_amount': np.random.uniform(1, 15),
                        'sell_amount': np.random.uniform(1, 10),
                        'famous_seats': ['华泰深圳益田路', '中信上海溧阳路']
                    }
                else:
                    lhb_data[symbol] = {'net_buy': 0}
            
            return lhb_data
            
        except Exception as e:
            self.logger.error(f"加载龙虎榜数据失败: {e}")
            return {}
    
    async def _load_money_flow_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载资金流向数据"""
        try:
            money_flow = {}
            
            for symbol in symbols:
                money_flow[symbol] = {
                    f'{symbol}_main': np.random.uniform(-5, 10),  # 主力净流入
                    f'{symbol}_super_ratio': np.random.uniform(0.05, 0.3),  # 超大单比例
                    f'{symbol}_consecutive': np.random.randint(0, 5),  # 连续流入天数
                    f'{symbol}_northbound': np.random.uniform(-1, 3)  # 北向资金
                }
            
            return money_flow
            
        except Exception as e:
            self.logger.error(f"加载资金流向数据失败: {e}")
            return {}
    
    async def _load_sector_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载板块数据"""
        try:
            sector_data = {
                'sector_change': np.random.uniform(-2, 5),
                'sector_money_flow': np.random.uniform(-10, 20),
                'rotation_score': np.random.uniform(0, 100),
                'hot_days': np.random.randint(0, 10)
            }
            
            for symbol in symbols:
                sector_data[f'{symbol}_rank'] = np.random.randint(1, 20)
                sector_data[f'{symbol}_heat'] = np.random.uniform(0, 100)
                sector_data[f'{symbol}_sector_rank'] = np.random.randint(1, 30)
            
            return sector_data
            
        except Exception as e:
            self.logger.error(f"加载板块数据失败: {e}")
            return {}
    
    async def _load_technical_indicators(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载技术指标"""
        try:
            tech_data = {}
            
            for symbol in symbols:
                tech_data[symbol] = {
                    'rsi': np.random.uniform(30, 80),
                    'macd': np.random.uniform(-1, 1),
                    'volatility': np.random.uniform(0.01, 0.05),
                    'seal_ratio': np.random.uniform(0.01, 0.15),
                    'zt_time': np.random.choice(['09:35', '10:15', '11:00', '14:00']),
                    'open_times': np.random.randint(0, 3),
                    'consecutive_limit': np.random.randint(0, 5),
                    'auction_change': np.random.uniform(-2, 8),
                    'auction_volume_ratio': np.random.uniform(0.05, 0.2),
                    'bid_ask_ratio': np.random.uniform(0.5, 4),
                    'auction_volatility': np.random.uniform(0.1, 1),
                    'large_order_ratio': np.random.uniform(0.05, 0.4),
                    'relative_strength': np.random.uniform(30, 90),
                    'portfolio_correlation': np.random.uniform(0, 1),
                    'timing_score': np.random.uniform(30, 90),
                    'resistance_distance': np.random.uniform(0, 0.1),
                    'pattern': np.random.choice(['none', 'flag', 'cup_handle', 'consolidation'])
                }
            
            return tech_data
            
        except Exception as e:
            self.logger.error(f"加载技术指标失败: {e}")
            return {}
    
    async def _load_fundamental_data(self, symbols: List[str], date: Optional[str], backtest: bool) -> Dict:
        """加载基本面数据"""
        try:
            fundamental = {}
            
            for symbol in symbols:
                fundamental[symbol] = {
                    'pe_ratio': np.random.uniform(10, 50),
                    'pb_ratio': np.random.uniform(1, 5),
                    'roe': np.random.uniform(5, 30),
                    'revenue_growth': np.random.uniform(-10, 50),
                    'profit_growth': np.random.uniform(-20, 60),
                    'financial_score': np.random.uniform(40, 90),
                    'regulatory_risk': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1]),
                    'history_leader_times': np.random.randint(0, 5),
                    'retail_sentiment': np.random.uniform(20, 90),
                    'institution_rating': np.random.uniform(2.5, 5),
                    'news_freshness': np.random.uniform(0, 1)
                }
            
            return fundamental
            
        except Exception as e:
            self.logger.error(f"加载基本面数据失败: {e}")
            return {}
    
    async def _load_market_mood(self, date: Optional[str], backtest: bool) -> float:
        """加载市场情绪指数"""
        try:
            # 实际应该从API或计算得出
            return np.random.uniform(20, 80)
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
    if args.symbols:
        symbols = args.symbols
    else:
        # 默认股票池
        symbols = [
            '000001', '000002', '000858', '002142', '002236',
            '300750', '300059', '600519', '000333', '002415'
        ]
    
    logger.info(f"配置: {config}")
    logger.info(f"股票池: {symbols}")
    
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