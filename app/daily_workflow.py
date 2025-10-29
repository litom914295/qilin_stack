"""
麒麟量化系统 - 每日自动化工作流
整合: 集合竞价监控 -> AI决策 -> 交易执行
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from app.auction_monitor_system import AuctionMonitor
from app.rl_decision_agent import RLDecisionAgent
from app.trading_executor import TradingExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'daily_workflow_{datetime.now():%Y%m%d}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DailyWorkflow:
    """每日自动化交易工作流"""
    
    def __init__(
        self, 
        account_balance: float = 100000,
        max_position_per_stock: float = 0.2,
        max_total_position: float = 0.9,
        top_n_stocks: int = 5,
        min_rl_score: float = 70.0,
        enable_real_trading: bool = False,
        use_neural_network: bool = False,
        test_mode: bool = False
    ):
        """
        初始化工作流
        
        Args:
            account_balance: 账户余额
            max_position_per_stock: 单股最大仓位
            max_total_position: 总仓位上限
            top_n_stocks: 最多买入股票数
            min_rl_score: 最低RL得分门槛
            enable_real_trading: 是否启用真实交易
            use_neural_network: 是否使用神经网络决策
        """
        self.config = {
            "account_balance": account_balance,
            "max_position_per_stock": max_position_per_stock,
            "max_total_position": max_total_position,
            "top_n_stocks": top_n_stocks,
            "min_rl_score": min_rl_score,
            "enable_real_trading": enable_real_trading,
            "use_neural_network": use_neural_network,
            "test_mode": test_mode
        }
        
        # 初始化各模块
        self.auction_monitor = AuctionMonitor(refresh_interval=3)
        self.test_mode = test_mode
        
        self.rl_agent = RLDecisionAgent(
            use_neural_network=use_neural_network,
            weights_path="config/rl_weights.json"
        )
        
        self.executor = TradingExecutor(
            account_balance=account_balance,
            max_position_per_stock=max_position_per_stock,
            max_total_position=max_total_position,
            enable_real_trading=enable_real_trading
        )
        
        self.workflow_report = {}
        
        logger.info("=" * 80)
        logger.info("麒麟量化系统 - 每日自动化工作流初始化完成")
        logger.info("=" * 80)
        logger.info(f"配置信息:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
    
    async def run_daily_workflow(self):
        """
        执行每日工作流
        
        流程:
        1. 9:15-9:26 监控昨日涨停板集合竞价
        2. 9:26 分析结束,生成竞价报告
        3. AI智能体决策,选出优质股票
        4. 9:30 开盘后执行买入
        5. 盘中监控,止盈止损
        6. 生成日报
        """
        logger.info("\n" + "🚀" * 40)
        logger.info(f"开始执行每日工作流 - {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info("🚀" * 40 + "\n")
        
        try:
            # ============ 步骤1: 集合竞价监控 ============
            logger.info("\n" + "=" * 60)
            logger.info("步骤1️⃣: 9:15-9:26 集合绞价实时监控")
            logger.info("=" * 60)
            
            # 自动判断是否在绞价时间
            now = datetime.now().time()
            is_auction_time = (now >= datetime.strptime("09:15", "%H:%M").time() and 
                              now <= datetime.strptime("09:26", "%H:%M").time())
            
            if self.test_mode and not is_auction_time:
                logger.warning("🧪 非绞价时间+测试模式: 使用模拟绞价数据")
                auction_report = self._generate_mock_auction_report()
            elif not is_auction_time:
                logger.error("❌ 当前不在绞价时间段(09:15-09:26)，请启用test_mode或在绞价时间运行")
                return None
            else:
                logger.info("✅ 绞价时间段，开始获取实时数据...")
                auction_report = await self.auction_monitor.start()
            
            if not auction_report or not auction_report.get("stocks"):
                logger.error("❌ 竞价监控失败或无有效数据,工作流中止")
                return None
            
            self.workflow_report["auction_report"] = auction_report
            logger.info(f"✅ 竞价监控完成,共分析 {len(auction_report['stocks'])} 只股票")
            
            # ============ 步骤2: AI智能体决策 ============
            logger.info("\n" + "=" * 60)
            logger.info("步骤2️⃣: 强化学习智能体选股决策")
            logger.info("=" * 60)
            
            ranked_stocks = self.rl_agent.rank_stocks(auction_report)
            
            selected_stocks = self.rl_agent.select_top_stocks(
                ranked_stocks,
                top_n=self.config["top_n_stocks"],
                min_score=self.config["min_rl_score"]
            )
            
            if not selected_stocks:
                logger.warning("⚠️  未选出符合条件的股票,工作流中止")
                return None
            
            self.workflow_report["ranked_stocks"] = ranked_stocks
            self.workflow_report["selected_stocks"] = selected_stocks
            
            logger.info(f"✅ AI决策完成,最终选中 {len(selected_stocks)} 只股票:")
            for i, stock in enumerate(selected_stocks, 1):
                logger.info(
                    f"  {i}. {stock['symbol']} {stock['name']} - "
                    f"RL得分 {stock['rl_score']:.2f}, "
                    f"连板 {stock['yesterday_info']['consecutive_days']}天"
                )
            
            # ============ 步骤3: 等待开盘 ============
            logger.info("\n" + "=" * 60)
            logger.info("步骤3️⃣: 等待9:30开盘...")
            logger.info("=" * 60)
            
            await self._wait_for_market_open()
            
            # ============ 步骤4: 执行买入 ============
            logger.info("\n" + "=" * 60)
            logger.info("步骤4️⃣: 开盘后批量买入")
            logger.info("=" * 60)
            
            buy_orders = self.executor.batch_buy(selected_stocks)
            self.workflow_report["buy_orders"] = [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "name": o.name,
                    "price": o.filled_price,
                    "volume": o.filled_volume,
                    "cost": o.filled_price * o.filled_volume
                }
                for o in buy_orders
            ]
            
            logger.info(f"✅ 批量买入完成,成功 {len(buy_orders)} 笔")
            
            # ============ 步骤5: 盘中监控 ============
            logger.info("\n" + "=" * 60)
            logger.info("步骤5️⃣: 盘中实时监控 (止盈止损)")
            logger.info("=" * 60)
            
            await self._intraday_monitor()
            
            # ============ 步骤6: 生成日报 ============
            logger.info("\n" + "=" * 60)
            logger.info("步骤6️⃣: 生成每日交易报告")
            logger.info("=" * 60)
            
            self._generate_daily_report()
            
            logger.info("\n" + "🎉" * 40)
            logger.info("每日工作流执行完成!")
            logger.info("🎉" * 40 + "\n")
            
            return self.workflow_report
            
        except Exception as e:
            logger.error(f"❌ 工作流执行失败: {e}", exc_info=True)
            return None
    
    async def _wait_for_market_open(self):
        """等待开盘"""
        now = datetime.now()
        market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now < market_open_time:
            wait_seconds = (market_open_time - now).seconds
            logger.info(f"⏳ 距离开盘还有 {wait_seconds} 秒...")
            
            # 实际应用中等待
            # await asyncio.sleep(wait_seconds)
            
            # 测试时跳过等待
            logger.info("测试模式: 跳过等待")
        else:
            logger.info("✅ 已开盘")
    
    async def _intraday_monitor(self):
        """
        盘中监控 - 实时更新持仓,执行止盈止损
        
        策略:
        1. 涨停立即卖出 (T+1限制,次日卖)
        2. 涨幅>=5% 卖出50%
        3. 跌幅>=3% 止损
        4. 尾盘检查,决定是否持仓过夜
        """
        logger.info("开始盘中监控...")
        
        # 模拟盘中监控 (实际应每分钟或每秒检查一次)
        monitor_rounds = 5  # 测试用
        
        for round_num in range(1, monitor_rounds + 1):
            logger.info(f"\n--- 监控轮次 {round_num}/{monitor_rounds} ---")
            
            # 获取当前持仓
            portfolio = self.executor.get_portfolio_status()
            
            if not portfolio["positions"]:
                logger.info("当前无持仓")
                break
            
            # 模拟获取实时价格 (实际应从行情接口获取)
            # TODO: 接入真实行情数据
            import numpy as np
            market_data = {}
            for symbol, pos in portfolio["positions"].items():
                # 模拟价格变化 (-5% ~ +12%)
                change = np.random.uniform(-0.05, 0.12)
                market_data[symbol] = pos["cost_price"] * (1 + change)
            
            # 更新持仓价格
            self.executor.update_positions_price(market_data)
            
            # 检查止盈止损
            for symbol, position in list(portfolio["positions"].items()):
                profit_rate = position["profit_rate"]
                
                if profit_rate >= 0.098:  # 接近涨停
                    logger.info(f"🎯 {symbol} 接近涨停 ({profit_rate:.2%}), 准备次日卖出")
                    # T+1限制,标记为次日卖出
                    position["sell_next_day"] = True
                
                elif profit_rate >= 0.05:  # 涨幅>=5%
                    logger.info(f"💰 {symbol} 涨幅 {profit_rate:.2%}, 卖出50%止盈")
                    sell_volume = position["volume"] // 2
                    if sell_volume >= 100:
                        self.executor.sell(
                            symbol=symbol,
                            reason=f"止盈 (涨幅{profit_rate:.2%})",
                            volume=sell_volume
                        )
                
                elif profit_rate <= -0.03:  # 跌幅>=3%
                    logger.warning(f"⚠️  {symbol} 跌幅 {profit_rate:.2%}, 全部卖出止损")
                    self.executor.sell(
                        symbol=symbol,
                        reason=f"止损 (跌幅{profit_rate:.2%})"
                    )
            
            # 每轮等待 (测试时缩短)
            await asyncio.sleep(1)  # 实际应为60秒或更长
        
        logger.info("盘中监控结束")
    
    def _generate_daily_report(self):
        """生成每日报告并保存为Dashboard所需的JSON"""
        import json
        from pathlib import Path
        
        portfolio = self.executor.get_portfolio_status()
        date_str = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存竞价报告
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        auction_report = self.workflow_report.get("auction_report")
        if auction_report:
            auction_file = reports_dir / f"auction_report_{date_str}_{timestamp}.json"
            with open(auction_file, 'w', encoding='utf-8') as f:
                json.dump(auction_report, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 竞价报告已保存: {auction_file}")
        
        # 保存RL决策结果
        rl_decision_data = {
            "date": date_str,
            "timestamp": timestamp,
            "config": self.config,
            "ranked_stocks": self.workflow_report.get("ranked_stocks", []),
            "selected_stocks": self.workflow_report.get("selected_stocks", [])
        }
        
        rl_file = reports_dir / f"rl_decision_{date_str}_{timestamp}.json"
        with open(rl_file, 'w', encoding='utf-8') as f:
            json.dump(rl_decision_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 RL决策结果已保存: {rl_file}")
        
        report = {
            "date": date_str,
            "config": self.config,
            "auction": {
                "monitored_stocks": len(self.workflow_report.get("auction_report", {}).get("stocks", [])),
                "top_stocks": [
                    {
                        "symbol": s["symbol"],
                        "name": s["name"],
                        "auction_strength": s["auction_info"]["strength"]
                    }
                    for s in self.workflow_report.get("auction_report", {}).get("stocks", [])[:5]
                ]
            },
            "decision": {
                "selected_count": len(self.workflow_report.get("selected_stocks", [])),
                "selected_stocks": [
                    {
                        "symbol": s["symbol"],
                        "name": s["name"],
                        "rl_score": s["rl_score"]
                    }
                    for s in self.workflow_report.get("selected_stocks", [])
                ]
            },
            "trading": {
                "buy_orders": len(self.workflow_report.get("buy_orders", [])),
                "total_buy_amount": sum(
                    o["cost"] for o in self.workflow_report.get("buy_orders", [])
                ),
                "sell_orders": len(self.executor.trade_history),
                "trades": self.executor.trade_history
            },
            "portfolio": portfolio,
            "performance": {
                "total_profit": portfolio["total_profit"],
                "profit_rate": portfolio["profit_rate"],
                "position_ratio": portfolio["position_ratio"]
            }
        }
        
        # 保存报告
        report_dir = Path("reports/daily")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"daily_report_{datetime.now():%Y%m%d}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📊 日报已保存: {report_file}")
        
        # 打印关键指标
        logger.info("\n" + "=" * 60)
        logger.info("📊 每日交易总结")
        logger.info("=" * 60)
        logger.info(f"监控股票数: {report['auction']['monitored_stocks']}")
        logger.info(f"选中股票数: {report['decision']['selected_count']}")
        logger.info(f"买入笔数: {report['trading']['buy_orders']}")
        logger.info(f"买入金额: {report['trading']['total_buy_amount']:.2f}")
        logger.info(f"卖出笔数: {report['trading']['sell_orders']}")
        logger.info(f"当前持仓数: {portfolio['position_count']}")
        logger.info(f"总盈亏: {portfolio['total_profit']:.2f} ({portfolio['profit_rate']:.2%})")
        logger.info(f"仓位比例: {portfolio['position_ratio']:.1%}")
        logger.info("=" * 60)
        
        self.workflow_report["daily_report"] = report
    
    def _generate_mock_auction_report(self):
        """生成模拟竞价报告(测试用)"""
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        mock_stocks = [
            {
                "symbol": "300750",
                "name": "宁德时代",
                "yesterday_info": {
                    "date": yesterday,
                    "consecutive_days": 1,
                    "seal_ratio": 0.08,
                    "is_first_board": True,
                    "quality_score": 92.0,
                    "sector": "新能源",
                    "is_sector_leader": True
                },
                "auction_info": {
                    "current_price": 245.5,
                    "change_pct": 0.023,
                    "strength": 85.0,
                    "volume": 120000,
                    "buy_sell_ratio": 2.3
                }
            },
            {
                "symbol": "600519",
                "name": "贵州茅台",
                "yesterday_info": {
                    "date": yesterday,
                    "consecutive_days": 3,
                    "seal_ratio": 0.20,
                    "is_first_board": False,
                    "quality_score": 88.0,
                    "sector": "消费",
                    "is_sector_leader": True
                },
                "auction_info": {
                    "current_price": 2025.0,
                    "change_pct": 0.015,
                    "strength": 78.0,
                    "volume": 85000,
                    "buy_sell_ratio": 1.9
                }
            },
            {
                "symbol": "000001",
                "name": "平安银行",
                "yesterday_info": {
                    "date": yesterday,
                    "consecutive_days": 2,
                    "seal_ratio": 0.15,
                    "is_first_board": False,
                    "quality_score": 85.0,
                    "sector": "金融",
                    "is_sector_leader": True
                },
                "auction_info": {
                    "current_price": 11.25,
                    "change_pct": 0.023,
                    "strength": 82.0,
                    "volume": 150000,
                    "buy_sell_ratio": 2.1
                }
            }
        ]
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "模拟绞价",
            "stocks": mock_stocks,
            "summary": {
                "total_count": len(mock_stocks),
                "avg_strength": sum(s["auction_info"]["strength"] for s in mock_stocks) / len(mock_stocks),
                "strong_count": sum(1 for s in mock_stocks if s["auction_info"]["strength"] >= 80)
            }
        }


async def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    # 创建工作流
    workflow = DailyWorkflow(
        account_balance=100000,
        max_position_per_stock=0.25,
        max_total_position=0.9,
        top_n_stocks=5,
        min_rl_score=70.0,
        enable_real_trading=False,  # 模拟交易
        use_neural_network=False,   # 使用加权打分
        test_mode=test_mode         # 通过命令行参数控制
    )
    
    if test_mode:
        logger.info("🧪 测试模式已启用，将使用模拟数据")
    else:
        logger.info("✅ 生产模式，将在绞价时间自动获取实时数据")
    
    # 运行工作流
    result = await workflow.run_daily_workflow()
    
    if result:
        logger.info("\n✅ 工作流执行成功!")
    else:
        logger.error("\n❌ 工作流执行失败!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  工作流已手动停止")
    except Exception as e:
        logger.error(f"\n\n❌ 工作流异常: {e}", exc_info=True)
