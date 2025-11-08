"""
策略优化闭环系统 - Qilin Stack 核心特色
===========================================

完整流程:
1. AI因子挖掘 (RD-Agent) → 生成因子和策略
2. 回测验证 (Qlib) → 评估策略表现
3. 模拟交易 (Live Trading) → 实盘前测试
4. 性能反馈 → 回传给AI优化
5. 迭代优化 → 持续改进

这是 Qilin Stack 的核心创新:
- Qlib: 提供回测引擎
- RD-Agent: 提供AI策略生成
- Qilin Stack: 建立完整闭环连接

Author: Qilin Stack Team
Date: 2024-11-08
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path

# 导入Qilin Stack核心模块
from rd_agent.compat_wrapper import RDAgentWrapper
from rd_agent.logging_integration import QilinRDAgentLogger
from app.core.backtest_engine import BacktestEngine, Order, OrderSide, OrderType
from trading.live_trading_system import create_live_trading_system

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """策略性能指标"""
    strategy_id: str
    strategy_name: str
    
    # 回测指标
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    
    # 因子指标
    ic_mean: float = 0.0
    icir: float = 0.0
    turnover: float = 0.0
    
    # 实盘指标 (模拟交易)
    live_pnl: float = 0.0
    live_sharpe: float = 0.0
    live_days: int = 0
    
    # 综合评分
    overall_score: float = 0.0
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    iteration: int = 0


@dataclass
class FeedbackSignal:
    """反馈信号 - 用于AI优化"""
    signal_type: str  # 'positive', 'negative', 'neutral'
    aspect: str  # 'return', 'risk', 'stability', 'ic'
    message: str
    value: float
    suggestion: str  # 给AI的优化建议


class StrategyFeedbackLoop:
    """
    策略优化闭环系统
    
    核心功能:
    1. 使用RD-Agent生成策略
    2. 回测验证策略
    3. 模拟交易测试
    4. 收集反馈信号
    5. 优化迭代
    """
    
    def __init__(self, 
                 rd_agent_config: Dict[str, Any],
                 backtest_config: Dict[str, Any],
                 live_config: Optional[Dict[str, Any]] = None,
                 workspace_path: str = "./strategy_loop"):
        """
        初始化闭环系统
        
        Args:
            rd_agent_config: RD-Agent配置
            backtest_config: 回测配置
            live_config: 实盘/模拟盘配置
            workspace_path: 工作目录
        """
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化各组件
        self.rd_agent = RDAgentWrapper(rd_agent_config)
        self.logger = QilinRDAgentLogger(str(self.workspace_path / 'logs'))
        self.backtest_engine = BacktestEngine(**backtest_config)
        
        # 模拟交易系统 (可选)
        self.live_system = None
        if live_config:
            self.live_system = create_live_trading_system(live_config)
        
        # 性能跟踪
        self.performance_history: List[StrategyPerformance] = []
        self.feedback_history: List[FeedbackSignal] = []
        self.current_iteration = 0
        
        logger.info("✅ 策略优化闭环系统已初始化")
    
    async def run_full_loop(self,
                           research_topic: str,
                           data: pd.DataFrame,
                           max_iterations: int = 5,
                           performance_threshold: float = 0.15) -> Dict[str, Any]:
        """
        运行完整的优化闭环
        
        Args:
            research_topic: 研究主题
            data: 历史数据
            max_iterations: 最大迭代次数
            performance_threshold: 性能阈值 (年化收益>15%)
        
        Returns:
            最优策略和性能报告
        """
        logger.info(f"🚀 开始策略优化闭环: {research_topic}")
        logger.info(f"   最大迭代: {max_iterations}次")
        logger.info(f"   目标收益: >{performance_threshold*100}%")
        
        best_strategy = None
        best_performance = None
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration + 1
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 第 {self.current_iteration}/{max_iterations} 轮迭代")
            logger.info(f"{'='*60}")
            
            # ========== 阶段1: AI因子挖掘 ==========
            logger.info("🤖 阶段1: AI因子挖掘...")
            
            # 构建包含反馈的研究主题
            enhanced_topic = self._enhance_topic_with_feedback(
                research_topic, 
                self.feedback_history
            )
            
            factors_result = await self.rd_agent.research_pipeline(
                research_topic=enhanced_topic,
                data=data,
                max_iterations=3
            )
            
            if not factors_result.get('factors'):
                logger.warning("⚠️ 未发现有效因子,跳过此轮")
                continue
            
            logger.info(f"✅ 发现 {len(factors_result['factors'])} 个因子")
            
            # ========== 阶段2: 策略构建 ==========
            logger.info("📊 阶段2: 构建交易策略...")
            
            strategy = self._build_strategy_from_factors(
                factors_result['factors']
            )
            
            # ========== 阶段3: 回测验证 ==========
            logger.info("⚡ 阶段3: 回测验证...")
            
            backtest_result = await self._run_backtest(strategy, data)
            
            # ========== 阶段4: 模拟交易 (可选) ==========
            live_result = None
            if self.live_system:
                logger.info("💼 阶段4: 模拟交易测试...")
                live_result = await self._run_live_test(strategy, data)
            
            # ========== 阶段5: 性能评估 ==========
            logger.info("📈 阶段5: 性能评估...")
            
            performance = self._calculate_performance(
                strategy,
                factors_result,
                backtest_result,
                live_result
            )
            
            self.performance_history.append(performance)
            
            # 记录到日志
            self.logger.log_experiment(
                {
                    'iteration': iteration,
                    'strategy': strategy,
                    'performance': performance.__dict__
                },
                tag=f'loop.{research_topic}'
            )
            
            logger.info(f"\n{'='*60}")
            logger.info("📊 本轮性能:")
            logger.info(f"   年化收益: {performance.annual_return*100:.2f}%")
            logger.info(f"   夏普比率: {performance.sharpe_ratio:.2f}")
            logger.info(f"   最大回撤: {performance.max_drawdown*100:.2f}%")
            logger.info(f"   IC均值: {performance.ic_mean:.4f}")
            logger.info(f"   综合得分: {performance.overall_score:.2f}/100")
            logger.info(f"{'='*60}\n")
            
            # ========== 阶段6: 生成反馈 ==========
            logger.info("🔍 阶段6: 生成优化反馈...")
            
            feedback_signals = self._generate_feedback(
                performance,
                backtest_result
            )
            
            self.feedback_history.extend(feedback_signals)
            
            for signal in feedback_signals:
                logger.info(f"   [{signal.signal_type.upper()}] {signal.aspect}: {signal.message}")
            
            # ========== 阶段7: 判断是否达标 ==========
            if performance.annual_return > performance_threshold:
                if best_performance is None or \
                   performance.overall_score > best_performance.overall_score:
                    best_strategy = strategy
                    best_performance = performance
                    
                    logger.info(f"✅ 发现更优策略! 综合得分: {performance.overall_score:.2f}")
                    
                    # 如果足够好,可以提前结束
                    if performance.overall_score > 85:
                        logger.info("🎉 发现优秀策略,提前结束优化")
                        break
            
            # 保存中间结果
            self._save_checkpoint(iteration, strategy, performance)
        
        # ========== 生成最终报告 ==========
        final_report = self._generate_final_report(
            best_strategy,
            best_performance,
            research_topic
        )
        
        logger.info(f"\n{'='*60}")
        logger.info("🎊 优化闭环完成!")
        logger.info(f"   总迭代次数: {self.current_iteration}")
        logger.info(f"   最优年化收益: {best_performance.annual_return*100:.2f}%")
        logger.info(f"   最优夏普: {best_performance.sharpe_ratio:.2f}")
        logger.info(f"   最优得分: {best_performance.overall_score:.2f}/100")
        logger.info(f"{'='*60}\n")
        
        return final_report
    
    def _enhance_topic_with_feedback(self,
                                     topic: str,
                                     feedback: List[FeedbackSignal]) -> str:
        """
        使用反馈信号增强研究主题
        
        这是闭环的关键: 将上一轮的问题告诉AI
        """
        if not feedback:
            return topic
        
        # 获取最近的反馈
        recent_feedback = feedback[-5:]  # 最近5条
        
        suggestions = []
        for signal in recent_feedback:
            if signal.signal_type == 'negative':
                suggestions.append(signal.suggestion)
        
        if suggestions:
            enhanced = f"{topic}\n\n优化建议:\n"
            enhanced += "\n".join(f"- {s}" for s in suggestions)
            return enhanced
        
        return topic
    
    def _build_strategy_from_factors(self, factors: List) -> Dict[str, Any]:
        """
        从因子构建交易策略
        
        策略包含:
        - 因子组合
        - 权重分配
        - 交易规则
        """
        strategy = {
            'name': f'AI_Strategy_{self.current_iteration}',
            'factors': [],
            'weights': [],
            'rules': {
                'rebalance_frequency': 'weekly',  # 每周调仓
                'top_k': 30,  # 买入前30只
                'position_limit': 0.1,  # 单只股票最多10%
                'stop_loss': -0.05,  # 止损5%
                'take_profit': 0.15  # 止盈15%
            }
        }
        
        # 提取因子
        for factor in factors:
            strategy['factors'].append({
                'name': factor.name,
                'expression': factor.expression,
                'ic': factor.performance.get('ic', 0)
            })
            
            # 根据IC分配权重
            ic = abs(factor.performance.get('ic', 0))
            strategy['weights'].append(ic)
        
        # 归一化权重
        total_weight = sum(strategy['weights'])
        if total_weight > 0:
            strategy['weights'] = [w/total_weight for w in strategy['weights']]
        else:
            # 均分权重
            n = len(strategy['factors'])
            strategy['weights'] = [1.0/n] * n
        
        return strategy
    
    async def _run_backtest(self,
                           strategy: Dict[str, Any],
                           data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行回测
        
        Returns:
            回测结果 (收益曲线, 交易记录等)
        """
        # 设置数据
        self.backtest_engine.set_data(data)
        
        # 计算因子信号
        signals = self._calculate_factor_signals(strategy, data)
        
        # 模拟交易
        for date, signal_data in signals.iterrows():
            self.backtest_engine.current_timestamp = date
            
            # 解冻持仓 (T+1)
            self.backtest_engine.portfolio.unfreeze_positions(date)
            
            # 根据信号生成订单
            top_stocks = signal_data.nlargest(strategy['rules']['top_k'])
            
            for symbol, score in top_stocks.items():
                if score > 0:
                    # 买入信号
                    # 计算买入数量 (等权重)
                    target_value = (
                        self.backtest_engine.portfolio.get_total_value() * 
                        strategy['rules']['position_limit']
                    )
                    
                    current_price = self._get_price(data, symbol, date)
                    if current_price > 0:
                        quantity = int(target_value / current_price / 100) * 100  # 100股整数倍
                        
                        if quantity > 0:
                            order = Order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=quantity
                            )
                            self.backtest_engine.place_order(order)
            
            # 止损/止盈检查
            for symbol, position in list(self.backtest_engine.portfolio.positions.items()):
                pnl_pct = position.unrealized_pnl / position.cost_basis
                
                # 止损
                if pnl_pct < strategy['rules']['stop_loss']:
                    if position.available_quantity > 0:
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=position.available_quantity
                        )
                        self.backtest_engine.place_order(order)
                
                # 止盈
                elif pnl_pct > strategy['rules']['take_profit']:
                    if position.available_quantity > 0:
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=position.available_quantity
                        )
                        self.backtest_engine.place_order(order)
        
        # 计算回测结果
        result = {
            'equity_curve': self.backtest_engine.portfolio.equity_curve,
            'trades': self.backtest_engine.portfolio.trades,
            'final_value': self.backtest_engine.portfolio.get_total_value(),
            'returns': self.backtest_engine.portfolio.get_returns()
        }
        
        return result
    
    def _calculate_factor_signals(self,
                                  strategy: Dict[str, Any],
                                  data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子信号
        
        Returns:
            每日每只股票的综合得分
        """
        # 简化版: 使用因子表达式计算
        # 实际应使用Qlib的Alpha表达式引擎
        
        signals = pd.DataFrame(index=data.index)
        
        for i, factor in enumerate(strategy['factors']):
            weight = strategy['weights'][i]
            
            # 这里简化处理,实际应解析factor['expression']
            # 示例: 使用收益率作为信号
            factor_score = data.pct_change(20)  # 20日收益率
            signals[f'factor_{i}'] = factor_score * weight
        
        # 综合得分
        composite_signal = signals.sum(axis=1)
        
        return composite_signal
    
    def _get_price(self, data: pd.DataFrame, symbol: str, date: datetime) -> float:
        """获取指定日期的价格"""
        try:
            # 简化版: 假设data是单股票数据
            price = data.loc[date, 'close']
            return price
        except:
            return 0.0
    
    async def _run_live_test(self,
                            strategy: Dict[str, Any],
                            data: pd.DataFrame) -> Dict[str, Any]:
        """
        运行模拟交易测试
        
        使用最近的数据进行模拟
        """
        # 实现模拟交易逻辑
        # 这里简化,实际应连接live_trading_system
        
        return {
            'live_pnl': 0.0,
            'live_sharpe': 0.0,
            'live_days': 0
        }
    
    def _calculate_performance(self,
                               strategy: Dict[str, Any],
                               factors_result: Dict[str, Any],
                               backtest_result: Dict[str, Any],
                               live_result: Optional[Dict[str, Any]]) -> StrategyPerformance:
        """
        计算综合性能
        """
        # 回测指标
        returns_series = pd.Series(backtest_result.get('returns', [0]))
        annual_return = returns_series.mean() * 252 if len(returns_series) > 0 else 0
        sharpe = (returns_series.mean() / returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0
        
        equity_curve = pd.Series([e[1] for e in backtest_result.get('equity_curve', [])])
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # 因子指标
        factors = factors_result.get('factors', [])
        ic_values = [f.performance.get('ic', 0) for f in factors if f.performance]
        ic_mean = np.mean(ic_values) if ic_values else 0
        
        # 计算综合得分
        score = 0
        score += min(annual_return * 100, 40)  # 收益 40分
        score += min(sharpe * 10, 30)  # 夏普 30分
        score += max(20 - max_dd * 100, 0)  # 回撤 20分
        score += min(abs(ic_mean) * 100, 10)  # IC 10分
        
        performance = StrategyPerformance(
            strategy_id=strategy['name'],
            strategy_name=strategy['name'],
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=len(backtest_result.get('trades', [])),
            ic_mean=ic_mean,
            overall_score=score,
            iteration=self.current_iteration
        )
        
        # 实盘指标
        if live_result:
            performance.live_pnl = live_result.get('live_pnl', 0)
            performance.live_sharpe = live_result.get('live_sharpe', 0)
            performance.live_days = live_result.get('live_days', 0)
        
        return performance
    
    def _generate_feedback(self,
                          performance: StrategyPerformance,
                          backtest_result: Dict[str, Any]) -> List[FeedbackSignal]:
        """
        生成反馈信号
        
        这是闭环的关键: 分析问题,给出建议
        """
        feedback = []
        
        # 1. 收益反馈
        if performance.annual_return < 0.10:
            feedback.append(FeedbackSignal(
                signal_type='negative',
                aspect='return',
                message=f'收益率偏低 ({performance.annual_return*100:.2f}%)',
                value=performance.annual_return,
                suggestion='尝试更激进的因子,如动量、反转等'
            ))
        elif performance.annual_return > 0.20:
            feedback.append(FeedbackSignal(
                signal_type='positive',
                aspect='return',
                message=f'收益率优秀 ({performance.annual_return*100:.2f}%)',
                value=performance.annual_return,
                suggestion='保持当前因子方向'
            ))
        
        # 2. 风险反馈
        if performance.sharpe_ratio < 1.0:
            feedback.append(FeedbackSignal(
                signal_type='negative',
                aspect='risk',
                message=f'夏普比率偏低 ({performance.sharpe_ratio:.2f})',
                value=performance.sharpe_ratio,
                suggestion='增加风险控制,考虑波动率因子'
            ))
        
        if performance.max_drawdown > 0.25:
            feedback.append(FeedbackSignal(
                signal_type='negative',
                aspect='risk',
                message=f'回撤过大 ({performance.max_drawdown*100:.2f}%)',
                value=performance.max_drawdown,
                suggestion='加强止损策略,降低仓位'
            ))
        
        # 3. 因子质量反馈
        if abs(performance.ic_mean) < 0.03:
            feedback.append(FeedbackSignal(
                signal_type='negative',
                aspect='ic',
                message=f'IC值偏低 ({performance.ic_mean:.4f})',
                value=performance.ic_mean,
                suggestion='探索新的因子维度,如基本面、情绪等'
            ))
        
        # 4. 稳定性反馈
        if len(self.performance_history) > 1:
            prev_performance = self.performance_history[-2]
            return_change = abs(performance.annual_return - prev_performance.annual_return)
            
            if return_change > 0.10:
                feedback.append(FeedbackSignal(
                    signal_type='negative',
                    aspect='stability',
                    message=f'策略不稳定,收益波动大 ({return_change*100:.2f}%)',
                    value=return_change,
                    suggestion='寻找更稳健的因子组合'
                ))
        
        return feedback
    
    def _save_checkpoint(self,
                        iteration: int,
                        strategy: Dict[str, Any],
                        performance: StrategyPerformance):
        """保存检查点"""
        checkpoint_dir = self.workspace_path / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'strategy': strategy,
            'performance': performance.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = checkpoint_dir / f'checkpoint_{iteration}.json'
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)
        
        logger.debug(f"💾 检查点已保存: {checkpoint_file}")
    
    def _generate_final_report(self,
                              best_strategy: Dict[str, Any],
                              best_performance: StrategyPerformance,
                              research_topic: str) -> Dict[str, Any]:
        """生成最终报告"""
        report = {
            'research_topic': research_topic,
            'total_iterations': self.current_iteration,
            'best_strategy': best_strategy,
            'best_performance': best_performance.__dict__,
            'performance_history': [p.__dict__ for p in self.performance_history],
            'improvement': {
                'return': (
                    best_performance.annual_return - self.performance_history[0].annual_return
                    if self.performance_history else 0
                ),
                'sharpe': (
                    best_performance.sharpe_ratio - self.performance_history[0].sharpe_ratio
                    if self.performance_history else 0
                )
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告
        report_file = self.workspace_path / 'final_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 最终报告已保存: {report_file}")
        
        return report


# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """使用示例"""
    
    # 1. 配置
    rd_agent_config = {
        'llm_model': 'gpt-4',
        'llm_api_key': 'your-api-key',
        'max_iterations': 5,
        'workspace_path': './logs/rdagent'
    }
    
    backtest_config = {
        'initial_capital': 1000000,
        'commission_rate': 0.0003,
        'slippage_rate': 0.0001
    }
    
    live_config = {
        'broker_name': 'mock',
        'initial_cash': 100000,
        'risk_config': {
            'max_position': 0.1,
            'stop_loss': -0.05
        }
    }
    
    # 2. 创建闭环系统
    loop_system = StrategyFeedbackLoop(
        rd_agent_config=rd_agent_config,
        backtest_config=backtest_config,
        live_config=live_config,
        workspace_path='./strategy_loop'
    )
    
    # 3. 准备数据
    # 这里应该加载真实的股票数据
    import pandas as pd
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2024-01-01'),
        'close': np.random.randn(1461).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 1461)
    }).set_index('date')
    
    # 4. 运行闭环优化
    result = await loop_system.run_full_loop(
        research_topic="寻找A股短期动量因子",
        data=data,
        max_iterations=5,
        performance_threshold=0.15
    )
    
    print("\n🎉 优化完成!")
    print(f"最优年化收益: {result['best_performance']['annual_return']*100:.2f}%")
    print(f"最优夏普比率: {result['best_performance']['sharpe_ratio']:.2f}")
    print(f"收益提升: {result['improvement']['return']*100:.2f}%")


if __name__ == '__main__':
    import asyncio
    asyncio.run(example_usage())
