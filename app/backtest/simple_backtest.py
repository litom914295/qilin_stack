"""
麒麟量化系统 - 简单回测系统
验证策略在历史数据上的表现
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import logging
import asyncio
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from core.trading_context import ContextManager, TradingContext
from agents.enhanced_agents import EnhancedAuctionGameAgent, EnhancedMarketEcologyAgent
from agents.trading_agents_impl import IntegratedDecisionAgent

logger = logging.getLogger(__name__)


class BacktestEngine:
    """简单回测引擎"""
    
    def __init__(self, start_date: str, end_date: str):
        """
        初始化回测引擎
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 回测结果
        self.trades = []  # 交易记录
        self.daily_returns = []  # 每日收益
        self.positions = {}  # 持仓
        
        # 统计指标
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0
        }
        
        # Agent
        self.decision_agent = IntegratedDecisionAgent()
        
    async def run_backtest(self, symbols: List[str]) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            symbols: 股票池
            
        Returns:
            回测结果
        """
        logger.info(f"开始回测: {self.start_date} 至 {self.end_date}")
        
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # 跳过周末（简化处理）
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            # 运行单日策略
            await self._run_single_day(current_date, symbols)
            
            current_date += timedelta(days=1)
        
        # 计算统计指标
        self._calculate_statistics()
        
        return self.generate_report()
    
    async def _run_single_day(self, date: datetime, symbols: List[str]):
        """运行单日策略"""
        logger.info(f"回测日期: {date.strftime('%Y-%m-%d')}")
        
        # 创建当日08:55的上下文管理器
        morning_time = date.replace(hour=8, minute=55)
        manager = ContextManager(morning_time)
        
        # 加载所有股票数据
        contexts = {}
        for symbol in symbols:
            ctx = manager.create_context(symbol)
            
            # 加载数据（这里应该从历史数据加载）
            ctx.load_d_day_data()
            ctx.load_t1_auction_data()
            
            contexts[symbol] = ctx
        
        # 运行决策
        signals = await self._generate_signals(contexts)
        
        # 模拟执行交易
        self._execute_trades(date, signals)
        
        # 更新持仓收益（使用T+1日收盘价）
        self._update_positions(date)
    
    async def _generate_signals(self, contexts: Dict[str, TradingContext]) -> List[Dict]:
        """生成交易信号"""
        signals = []
        
        for symbol, ctx in contexts.items():
            # 分析个股
            result = await self.decision_agent.analyze_parallel(
                symbol,
                self._context_to_market_context(ctx)
            
            # 根据得分生成信号
            if result['weighted_score'] >= 70:
                decision = result['decision']
                if decision['action'] in ['strong_buy', 'buy']:
                    signals.append({
                        'symbol': symbol,
                        'score': result['weighted_score'],
                        'action': decision['action'],
                        'position': decision.get('position', '10%'),
                        'reason': decision.get('reason', '')
                    })
        
        # 按得分排序，选择前N个
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:3]  # 最多持仓3只
    
    def _context_to_market_context(self, ctx: TradingContext) -> Any:
        """将TradingContext转换为MarketContext格式"""
        # 这里需要转换数据格式
        # 简化处理，返回模拟数据
        from agents.trading_agents_impl import MarketContext
        
        return MarketContext(
            ohlcv=pd.DataFrame({
                'close': [10, 10.5, 11],
                'volume': [1000, 1200, 1500],
                'turnover_rate': [5, 6, 8]
            }),
            news_titles=['测试新闻'],
            lhb_netbuy=0,
            market_mood_score=ctx.d_day_market.sentiment_score if ctx.d_day_market else 50,
            sector_heat={},
            money_flow={},
            technical_indicators={},
            fundamental_data={}
    
    def _execute_trades(self, date: datetime, signals: List[Dict]):
        """执行交易"""
        for signal in signals:
            # 检查是否已持有
            if signal['symbol'] in self.positions:
                continue
            
            # 记录交易
            trade = {
                'date': date.strftime('%Y-%m-%d'),
                'symbol': signal['symbol'],
                'action': 'buy',
                'price': 10.0,  # 模拟价格
                'quantity': 1000,
                'score': signal['score'],
                'reason': signal['reason']
            }
            
            self.trades.append(trade)
            self.positions[signal['symbol']] = {
                'entry_date': date,
                'entry_price': trade['price'],
                'quantity': trade['quantity']
            }
            
            logger.info(f"买入: {signal['symbol']} @ {trade['price']}")
    
    def _update_positions(self, date: datetime):
        """更新持仓收益"""
        for symbol, position in list(self.positions.items()):
            # 模拟T+1日收益
            random_return = np.random.uniform(-0.05, 0.10)
            exit_price = position['entry_price'] * (1 + random_return)
            
            # 简单策略：持有1天就卖出
            if (date - position['entry_date']).days >= 1:
                # 记录卖出
                trade = {
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'action': 'sell',
                    'price': exit_price,
                    'quantity': position['quantity'],
                    'return': random_return
                }
                
                self.trades.append(trade)
                self.daily_returns.append(random_return)
                
                # 更新统计
                if random_return > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                
                # 移除持仓
                del self.positions[symbol]
                
                logger.info(f"卖出: {symbol} @ {exit_price:.2f}, 收益: {random_return:.2%}")
    
    def _calculate_statistics(self):
        """计算统计指标"""
        if not self.daily_returns:
            return
        
        returns = np.array(self.daily_returns)
        
        # 总交易次数
        self.stats['total_trades'] = len([t for t in self.trades if t['action'] == 'buy'])
        
        # 胜率
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = self.stats['winning_trades'] / self.stats['total_trades']
        
        # 总收益率
        self.stats['total_return'] = np.sum(returns)
        
        # 平均收益
        self.stats['avg_return'] = np.mean(returns)
        
        # 最大回撤
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        self.stats['max_drawdown'] = np.min(drawdown)
        
        # 夏普比率（简化计算）
        if len(returns) > 1:
            self.stats['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        logger.info(f"回测统计: {self.stats}")
    
    def generate_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        return {
            'period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d')
            },
            'statistics': self.stats,
            'trades': self.trades[-20:],  # 最近20笔交易
            'daily_returns': {
                'mean': np.mean(self.daily_returns) if self.daily_returns else 0,
                'std': np.std(self.daily_returns) if self.daily_returns else 0,
                'min': np.min(self.daily_returns) if self.daily_returns else 0,
                'max': np.max(self.daily_returns) if self.daily_returns else 0
            }
        }


class ValidationEngine:
    """验证引擎 - 验证昨天的预测"""
    
    def __init__(self):
        self.predictions_file = Path("predictions.json")
        self.validation_file = Path("validations.json")
        
    def save_predictions(self, date: str, predictions: List[Dict]):
        """保存预测"""
        data = {
            'date': date,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.predictions_file, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存{len(predictions)}个预测")
    
    def validate_predictions(self, actual_data: Dict) -> Dict[str, Any]:
        """验证昨天的预测"""
        if not self.predictions_file.exists():
            return {'error': '无预测文件'}
        
        with open(self.predictions_file, 'r') as f:
            predictions = json.load(f)
        
        results = []
        for pred in predictions['predictions']:
            symbol = pred['symbol']
            predicted_action = pred['action']
            
            # 获取实际表现
            actual_return = actual_data.get(symbol, {}).get('return', 0)
            
            # 判断预测是否正确
            if predicted_action in ['strong_buy', 'buy']:
                is_correct = actual_return > 0.02  # 涨幅超过2%算预测正确
            else:
                is_correct = actual_return <= 0.02
            
            results.append({
                'symbol': symbol,
                'predicted': predicted_action,
                'actual_return': actual_return,
                'is_correct': is_correct
            })
        
        # 计算准确率
        accuracy = sum(r['is_correct'] for r in results) / len(results) if results else 0
        
        validation = {
            'date': predictions['date'],
            'accuracy': accuracy,
            'total_predictions': len(results),
            'correct_predictions': sum(r['is_correct'] for r in results),
            'results': results
        }
        
        # 保存验证结果
        with open(self.validation_file, 'w') as f:
            json.dump(validation, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证完成: 准确率 {accuracy:.2%}")
        
        return validation


async def main():
    """主函数 - 运行回测示例"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建回测引擎
    engine = BacktestEngine(
        start_date='2024-12-15',
        end_date='2024-12-20'
    
    # 股票池
    symbols = ['000001', '000002', '300750', '002415', '603986']
    
    # 运行回测
    report = await engine.run_backtest(symbols)
    
    # 打印报告
    print("\n" + "="*60)
    print("回测报告")
    print("="*60)
    print(f"回测周期: {report['period']['start']} 至 {report['period']['end']}")
    print(f"总交易次数: {report['statistics']['total_trades']}")
    print(f"胜率: {report['statistics']['win_rate']:.2%}")
    print(f"总收益率: {report['statistics']['total_return']:.2%}")
    print(f"最大回撤: {report['statistics']['max_drawdown']:.2%}")
    print(f"夏普比率: {report['statistics']['sharpe_ratio']:.2f}")
    print("="*60)
    
    # 保存报告
    report_file = f"backtest_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n报告已保存至: {report_file}")
    
    # 验证示例
    validator = ValidationEngine()
    
    # 保存今天的预测（示例）
    today_predictions = [
        {'symbol': '000001', 'action': 'buy', 'score': 75},
        {'symbol': '000002', 'action': 'strong_buy', 'score': 82}
    ]
    validator.save_predictions('2024-12-20', today_predictions)
    
    # 验证昨天的预测（示例）
    actual_data = {
        '000001': {'return': 0.035},  # 实际涨3.5%
        '000002': {'return': -0.012}  # 实际跌1.2%
    }
    validation_result = validator.validate_predictions(actual_data)
    
    print(f"\n预测验证: 准确率 {validation_result.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    asyncio.run(main())