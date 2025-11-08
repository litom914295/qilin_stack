"""
统一工作流编排器
管理T日筛选、T+1竞价监控、买入执行、T+2卖出等全流程
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import logging
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

# 导入核心模块
try:
    from app.auction_decision_engine import AuctionDecisionEngine
    from features.auction_features import AuctionFeatureExtractor
    from strategies.layered_buy_strategy import LayeredBuyStrategy
    from strategies.t2_sell_strategy import T2SellStrategy
    from risk.kelly_position_manager import KellyPositionManager
    from risk.market_circuit_breaker import MarketCircuitBreaker
    from analysis.trading_journal import TradingJournal, TradeRecord
except Exception as e:
    logging.warning(f"部分模块导入失败: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """工作流阶段"""
    T_DAY_SCREENING = "T日筛选"
    T1_AUCTION_MONITOR = "T+1竞价监控"
    T1_BUY_EXECUTION = "T+1买入执行"
    T1_POSITION_MONITOR = "T+1持仓监控"
    T2_SELL_EXECUTION = "T+2卖出执行"
    POST_TRADE_ANALYSIS = "交易后分析"


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "待执行"
    RUNNING = "执行中"
    COMPLETED = "已完成"
    FAILED = "失败"
    SKIPPED = "跳过"


@dataclass
class WorkflowContext:
    """工作流上下文"""
    date: str
    stage: WorkflowStage
    status: WorkflowStatus
    data: Dict
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class TradingWorkflow:
    """
    交易工作流编排器
    
    完整流程：
    1. T日收盘后：筛选候选股票
    2. T+1日竞价：监控竞价数据，生成买入信号
    3. T+1日开盘：执行买入订单
    4. T+1日盘中：持仓监控
    5. T+2日开盘：生成卖出信号并执行
    6. 交易后：记录日志，复盘分析
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化工作流
        
        Parameters:
        -----------
        config: Dict
            工作流配置
        """
        self.config = config or self._default_config()
        self.context_history = []
        
        # 初始化核心组件
        self._init_components()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'enable_t_day_screening': True,
            'enable_t1_auction_monitor': True,
            'enable_t1_buy': True,
            'enable_t2_sell': True,
            'enable_journal': True,
            'enable_market_breaker': True,
            'enable_kelly_position': True,
            
            # 筛选参数
            'screening': {
                'min_seal_strength': 3.0,
                'min_prediction_score': 0.6,
                'max_candidates': 30
            },
            
            # 竞价参数
            'auction': {
                'min_auction_strength': 0.6,
                'monitor_start_time': '09:15',
                'monitor_end_time': '09:25'
            },
            
            # 买入参数
            'buy': {
                'max_position_per_stock': 0.10,
                'total_capital': 1000000
            },
            
            # 卖出参数
            'sell': {
                'enable_partial_sell': True
            },
            
            # 风控参数
            'risk': {
                'enable_breaker': True,
                'enable_kelly': True
            }
        }
    
    def _init_components(self):
        """初始化组件"""
        try:
            self.decision_engine = AuctionDecisionEngine()
            logger.info("✓ 决策引擎初始化完成")
        except Exception as e:
            logger.warning(f"决策引擎初始化失败: {e}")
            self.decision_engine = None
        
        try:
            self.feature_extractor = AuctionFeatureExtractor()
            logger.info("✓ 特征提取器初始化完成")
        except Exception as e:
            logger.warning(f"特征提取器初始化失败: {e}")
            self.feature_extractor = None
        
        try:
            self.buy_strategy = LayeredBuyStrategy()
            logger.info("✓ 买入策略初始化完成")
        except Exception as e:
            logger.warning(f"买入策略初始化失败: {e}")
            self.buy_strategy = None
        
        try:
            self.sell_strategy = T2SellStrategy()
            logger.info("✓ 卖出策略初始化完成")
        except Exception as e:
            logger.warning(f"卖出策略初始化失败: {e}")
            self.sell_strategy = None
        
        if self.config['enable_kelly_position']:
            try:
                self.position_manager = KellyPositionManager(
                    total_capital=self.config['buy']['total_capital']
                )
                logger.info("✓ Kelly仓位管理器初始化完成")
            except Exception as e:
                logger.warning(f"Kelly仓位管理器初始化失败: {e}")
                self.position_manager = None
        else:
            self.position_manager = None
        
        if self.config['enable_market_breaker']:
            try:
                self.market_breaker = MarketCircuitBreaker()
                logger.info("✓ 市场熔断器初始化完成")
            except Exception as e:
                logger.warning(f"市场熔断器初始化失败: {e}")
                self.market_breaker = None
        else:
            self.market_breaker = None
        
        if self.config['enable_journal']:
            try:
                self.journal = TradingJournal()
                logger.info("✓ 交易日志初始化完成")
            except Exception as e:
                logger.warning(f"交易日志初始化失败: {e}")
                self.journal = None
        else:
            self.journal = None
    
    def run_full_workflow(self, date: str) -> Dict:
        """
        运行完整工作流
        
        Parameters:
        -----------
        date: str
            交易日期
            
        Returns:
        --------
        Dict: 执行结果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"开始执行交易工作流 - {date}")
        logger.info(f"{'='*80}\n")
        
        workflow_result = {
            'date': date,
            'stages': {},
            'overall_status': 'success'
        }
        
        try:
            # Stage 1: T日筛选
            if self.config['enable_t_day_screening']:
                t_day_result = self.stage_t_day_screening(date)
                workflow_result['stages']['t_day_screening'] = t_day_result
                
                if t_day_result['status'] == 'failed':
                    workflow_result['overall_status'] = 'failed'
                    return workflow_result
            
            # Stage 2: T+1竞价监控
            if self.config['enable_t1_auction_monitor']:
                t1_auction_result = self.stage_t1_auction_monitor(date)
                workflow_result['stages']['t1_auction_monitor'] = t1_auction_result
                
                if t1_auction_result['status'] == 'failed':
                    workflow_result['overall_status'] = 'partial'
            
            # Stage 3: T+1买入执行
            if self.config['enable_t1_buy']:
                t1_buy_result = self.stage_t1_buy_execution(date)
                workflow_result['stages']['t1_buy_execution'] = t1_buy_result
            
            # Stage 4: T+2卖出执行
            if self.config['enable_t2_sell']:
                t2_sell_result = self.stage_t2_sell_execution(date)
                workflow_result['stages']['t2_sell_execution'] = t2_sell_result
            
            # Stage 5: 交易后分析
            post_analysis_result = self.stage_post_trade_analysis(date)
            workflow_result['stages']['post_analysis'] = post_analysis_result
            
            logger.info(f"\n{'='*80}")
            logger.info(f"工作流执行完成 - 状态: {workflow_result['overall_status']}")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            workflow_result['overall_status'] = 'failed'
            workflow_result['error'] = str(e)
        
        return workflow_result
    
    def stage_t_day_screening(self, date: str) -> Dict:
        """
        Stage 1: T日候选筛选
        
        在T日收盘后执行，筛选次日竞价候选股票
        """
        logger.info(f"\n{'─'*80}")
        logger.info(f"【Stage 1】T日候选筛选 - {date}")
        logger.info(f"{'─'*80}")
        
        context = WorkflowContext(
            date=date,
            stage=WorkflowStage.T_DAY_SCREENING,
            status=WorkflowStatus.RUNNING,
            data={},
            start_time=datetime.now()
        )
        
        try:
            # 1. 检查市场环境
            if self.market_breaker:
                market_signal = self._check_market_condition()
                if not market_signal.allow_new_positions:
                    logger.warning("⚠️  市场熔断，跳过候选筛选")
                    context.status = WorkflowStatus.SKIPPED
                    context.data['reason'] = '市场熔断'
                    return self._finalize_context(context)
            
            # 2. 获取涨停股票（模拟）
            limitup_stocks = self._get_limitup_stocks(date)
            logger.info(f"获取到涨停股票: {len(limitup_stocks)} 只")
            
            # 3. 筛选候选
            candidates = self._filter_candidates(
                limitup_stocks,
                min_seal_strength=self.config['screening']['min_seal_strength'],
                min_prediction_score=self.config['screening']['min_prediction_score'],
                max_count=self.config['screening']['max_candidates']
            )
            
            logger.info(f"✓ 筛选完成: {len(candidates)} 只候选股票")
            
            # 4. Kelly仓位分配
            if self.position_manager and len(candidates) > 0:
                positions = self.position_manager.calculate_positions(
                    candidates,
                    historical_performance={}
                )
                context.data['positions'] = positions
            
            context.status = WorkflowStatus.COMPLETED
            context.data['candidates'] = candidates
            context.data['candidate_count'] = len(candidates)
            
        except Exception as e:
            logger.error(f"❌ T日筛选失败: {e}")
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
        
        return self._finalize_context(context)
    
    def stage_t1_auction_monitor(self, date: str) -> Dict:
        """
        Stage 2: T+1竞价监控
        
        在T+1日集合竞价时执行，监控竞价数据
        """
        logger.info(f"\n{'─'*80}")
        logger.info(f"【Stage 2】T+1竞价监控 - {date}")
        logger.info(f"{'─'*80}")
        
        context = WorkflowContext(
            date=date,
            stage=WorkflowStage.T1_AUCTION_MONITOR,
            status=WorkflowStatus.RUNNING,
            data={},
            start_time=datetime.now()
        )
        
        try:
            # 模拟竞价数据
            auction_data = self._get_auction_data(date)
            logger.info(f"获取竞价数据: {len(auction_data)} 只")
            
            # 生成买入信号
            buy_signals = self._generate_buy_signals(
                auction_data,
                min_strength=self.config['auction']['min_auction_strength']
            )
            
            logger.info(f"✓ 生成买入信号: {len(buy_signals)} 个")
            
            context.status = WorkflowStatus.COMPLETED
            context.data['auction_data'] = auction_data
            context.data['buy_signals'] = buy_signals
            context.data['signal_count'] = len(buy_signals)
            
        except Exception as e:
            logger.error(f"❌ 竞价监控失败: {e}")
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
        
        return self._finalize_context(context)
    
    def stage_t1_buy_execution(self, date: str) -> Dict:
        """
        Stage 3: T+1买入执行
        
        在T+1日开盘后执行买入订单
        """
        logger.info(f"\n{'─'*80}")
        logger.info(f"【Stage 3】T+1买入执行 - {date}")
        logger.info(f"{'─'*80}")
        
        context = WorkflowContext(
            date=date,
            stage=WorkflowStage.T1_BUY_EXECUTION,
            status=WorkflowStatus.RUNNING,
            data={},
            start_time=datetime.now()
        )
        
        try:
            # 模拟买入订单
            buy_orders = self._execute_buy_orders(date)
            
            logger.info(f"✓ 买入执行完成: {len(buy_orders)} 笔")
            
            context.status = WorkflowStatus.COMPLETED
            context.data['buy_orders'] = buy_orders
            context.data['order_count'] = len(buy_orders)
            
        except Exception as e:
            logger.error(f"❌ 买入执行失败: {e}")
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
        
        return self._finalize_context(context)
    
    def stage_t2_sell_execution(self, date: str) -> Dict:
        """
        Stage 4: T+2卖出执行
        
        在T+2日开盘后执行卖出订单
        """
        logger.info(f"\n{'─'*80}")
        logger.info(f"【Stage 4】T+2卖出执行 - {date}")
        logger.info(f"{'─'*80}")
        
        context = WorkflowContext(
            date=date,
            stage=WorkflowStage.T2_SELL_EXECUTION,
            status=WorkflowStatus.RUNNING,
            data={},
            start_time=datetime.now()
        )
        
        try:
            # 模拟卖出订单
            sell_orders = self._execute_sell_orders(date)
            
            logger.info(f"✓ 卖出执行完成: {len(sell_orders)} 笔")
            
            context.status = WorkflowStatus.COMPLETED
            context.data['sell_orders'] = sell_orders
            context.data['order_count'] = len(sell_orders)
            
        except Exception as e:
            logger.error(f"❌ 卖出执行失败: {e}")
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
        
        return self._finalize_context(context)
    
    def stage_post_trade_analysis(self, date: str) -> Dict:
        """
        Stage 5: 交易后分析
        
        记录交易日志，生成复盘报告
        """
        logger.info(f"\n{'─'*80}")
        logger.info(f"【Stage 5】交易后分析 - {date}")
        logger.info(f"{'─'*80}")
        
        context = WorkflowContext(
            date=date,
            stage=WorkflowStage.POST_TRADE_ANALYSIS,
            status=WorkflowStatus.RUNNING,
            data={},
            start_time=datetime.now()
        )
        
        try:
            # 生成日度报告
            daily_report = self._generate_daily_report(date)
            
            logger.info(f"✓ 日度报告生成完成")
            
            context.status = WorkflowStatus.COMPLETED
            context.data['daily_report'] = daily_report
            
        except Exception as e:
            logger.error(f"❌ 交易后分析失败: {e}")
            context.status = WorkflowStatus.FAILED
            context.error = str(e)
        
        return self._finalize_context(context)
    
    # ==================== 辅助方法 ====================
    
    def _check_market_condition(self) -> object:
        """检查市场环境"""
        # 模拟市场数据
        market_data = {
            'index_changes': {'sh': 0.5, 'sz': 0.3, 'cyb': 0.8},
            'limit_up_count': 80,
            'limit_down_count': 30,
            'total_stocks': 4800,
            'avg_turnover': 2.5,
            'northbound_flow': 30,
            'daily_pnl_ratio': 0.02,
            'continuous_loss_days': 0,
            'max_drawdown': -0.05
        }
        return self.market_breaker.check_market_condition(market_data)
    
    def _get_limitup_stocks(self, date: str) -> pd.DataFrame:
        """获取涨停股票（模拟）"""
        n = np.random.randint(50, 100)
        return pd.DataFrame({
            'symbol': [f'{i:06d}.SZ' for i in np.random.randint(1, 999999, n)],
            'name': [f'股票{i}' for i in range(n)],
            'close': np.random.uniform(10, 100, n),
            'seal_strength': np.random.uniform(1, 10, n),
            'prediction_score': np.random.uniform(0.3, 0.95, n)
        })
    
    def _filter_candidates(self, stocks: pd.DataFrame, **filters) -> pd.DataFrame:
        """筛选候选"""
        filtered = stocks[
            (stocks['seal_strength'] >= filters['min_seal_strength']) &
            (stocks['prediction_score'] >= filters['min_prediction_score'])
        ].sort_values('prediction_score', ascending=False).head(filters['max_count'])
        
        return filtered
    
    def _get_auction_data(self, date: str) -> pd.DataFrame:
        """获取竞价数据（模拟）"""
        n = 20
        return pd.DataFrame({
            'symbol': [f'{i:06d}.SZ' for i in range(n)],
            'auction_price': np.random.uniform(10, 100, n),
            'auction_strength': np.random.uniform(0.3, 0.95, n)
        })
    
    def _generate_buy_signals(self, auction_data: pd.DataFrame, min_strength: float) -> List:
        """生成买入信号"""
        signals = auction_data[auction_data['auction_strength'] >= min_strength]
        return signals.to_dict('records')
    
    def _execute_buy_orders(self, date: str) -> List:
        """执行买入订单（模拟）"""
        n = np.random.randint(5, 15)
        return [{'symbol': f'{i:06d}.SZ', 'price': np.random.uniform(10, 100), 'volume': 1000} 
                for i in range(n)]
    
    def _execute_sell_orders(self, date: str) -> List:
        """执行卖出订单（模拟）"""
        n = np.random.randint(3, 10)
        return [{'symbol': f'{i:06d}.SZ', 'price': np.random.uniform(10, 110), 'volume': 600} 
                for i in range(n)]
    
    def _generate_daily_report(self, date: str) -> Dict:
        """生成日度报告"""
        return {
            'date': date,
            'candidates': 23,
            'buy_orders': 12,
            'sell_orders': 8,
            'profit': 3240.50,
            'profit_rate': 2.54
        }
    
    def _finalize_context(self, context: WorkflowContext) -> Dict:
        """完成上下文"""
        context.end_time = datetime.now()
        duration = (context.end_time - context.start_time).total_seconds()
        
        result = {
            'stage': context.stage.value,
            'status': context.status.value,
            'duration': f"{duration:.2f}s",
            'data': context.data
        }
        
        if context.error:
            result['error'] = context.error
        
        self.context_history.append(context)
        
        return result


# 使用示例
if __name__ == "__main__":
    # 创建工作流
    workflow = TradingWorkflow()
    
    # 运行完整工作流
    result = workflow.run_full_workflow(date="2024-11-01")
    
    # 打印结果
    print(f"\n\n{'='*80}")
    print(f"工作流执行结果")
    print(f"{'='*80}")
    print(f"日期: {result['date']}")
    print(f"总体状态: {result['overall_status']}")
    print(f"\n各阶段执行情况:")
    for stage_name, stage_result in result['stages'].items():
        print(f"  - {stage_result['stage']}: {stage_result['status']} ({stage_result['duration']})")
    
    print(f"\n✅ 工作流测试完成！")
