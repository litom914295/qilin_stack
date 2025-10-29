"""
麒麟量化系统 - 9:15-9:26集合竞价实时监控系统
专注于昨日涨停板强势股的竞价期监控
"""

import asyncio
import time
import logging
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from app.enhanced_limitup_selector import EnhancedLimitUpSelector, LimitUpStock
from app.sector_theme_manager import SectorThemeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuctionPhase(Enum):
    """竞价阶段"""
    BEFORE = "未开始"
    PHASE1 = "9:15-9:20(可撤单)"
    PHASE2 = "9:20-9:25(不可撤单)"
    PHASE3 = "9:25-9:30(即将开盘)"
    AFTER = "已开盘"


@dataclass
class AuctionSnapshot:
    """竞价快照数据"""
    timestamp: str
    symbol: str
    price: float
    change_pct: float
    bid_volume: int
    ask_volume: int
    bid_orders: int
    ask_orders: int
    total_volume: int
    bid_prices: List[float]  # 买1-5价格
    ask_prices: List[float]  # 卖1-5价格
    large_buy_orders: int    # 大单买入笔数
    large_sell_orders: int   # 大单卖出笔数


# 使用 EnhancedLimitUpSelector 中的 LimitUpStock 数据结构
# 不再需要重复定义


class AuctionMonitor:
    """9:15-9:26集合竞价实时监控"""
    
    def __init__(self, refresh_interval: int = 3):
        """
        初始化监控器
        
        Args:
            refresh_interval: 刷新间隔(秒), 建议3秒
        """
        self.refresh_interval = refresh_interval
        self.yesterday_limits: List[LimitUpStock] = []  # 使用增强版数据结构
        self.auction_data: Dict[str, List[AuctionSnapshot]] = {}
        self.is_running = False
        
        # 关键时间点
        self.start_time = dt_time(9, 15, 0)
        self.critical_time = dt_time(9, 20, 0)  # 撤单截止
        self.end_time = dt_time(9, 26, 0)
        
        # 增强模块
        self.selector = EnhancedLimitUpSelector()
        self.sector_manager = SectorThemeManager()
        
        logger.info("集合竞价监控系统初始化完成(增强版)")
    
    def load_yesterday_limit_up_stocks(self) -> List[LimitUpStock]:
        """
        加载昨日涨停板优选强势股
        
        筛选标准:
        1. 昨日涨停
        2. 开板次数<=2次(优选一字板)
        3. 封单强度>=3%(排除弱封)
        4. 连板天数>=1(优选连板)
        5. 板块龙头优先
        6. 质量分>=70分
        """
        logger.info("=" * 60)
        logger.info("开始加载昨日涨停板优选股票...")
        
        # TODO: 实际数据源接入
        # 示例: 从数据库/API获取昨日涨停数据
        # yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        # df = get_limit_up_stocks(yesterday_date)
        
        # 模拟数据(实际使用时替换)
        # 使用增强版数据结构
        mock_stocks = [
            LimitUpStock(
                symbol="000001",
                name="平安银行",
                date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                limit_up_time="09:30",
                open_times=0,
                seal_ratio=0.15,
                is_one_word=True,
                consecutive_days=2,
                is_first_board=False,
                prev_limit_up=True,
                sector="金融",
                themes=["金融", "银行", "保险"],
                sector_limit_count=2,
                is_sector_leader=True,
                prev_close=10.0,
                open=11.0,
                high=11.0,
                low=10.9,
                close=11.0,
                limit_price=11.0,
                volume=1000000.0,
                amount=11000000.0,
                turnover_rate=8.5,
                volume_ratio=2.5,
                vwap_slope_morning=0.02,
                max_drawdown_morning=-0.005,
                afternoon_strength=0.003,
                quality_score=85.0,
                confidence=0.85
            ),
            LimitUpStock(
                symbol="300750",
                name="宁德时代",
                date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                limit_up_time="10:05",
                open_times=1,
                seal_ratio=0.08,
                is_one_word=False,
                consecutive_days=1,
                is_first_board=True,
                prev_limit_up=False,
                sector="新能源",
                themes=["新能源", "锂电池", "汽车零部件"],
                sector_limit_count=5,
                is_sector_leader=True,
                prev_close=200.0,
                open=220.0,
                high=240.0,
                low=218.0,
                close=240.0,
                limit_price=240.0,
                volume=5000000.0,
                amount=1150000000.0,
                turnover_rate=12.3,
                volume_ratio=3.2,
                vwap_slope_morning=0.05,
                max_drawdown_morning=-0.01,
                afternoon_strength=0.008,
                quality_score=92.0,
                confidence=0.90
            ),
            LimitUpStock(
                symbol="600519",
                name="贵州茅台",
                date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                limit_up_time="09:45",
                open_times=0,
                seal_ratio=0.20,
                is_one_word=False,
                consecutive_days=3,
                is_first_board=False,
                prev_limit_up=True,
                sector="消费",
                themes=["消费", "白酒", "食品饮料"],
                sector_limit_count=1,
                is_sector_leader=True,
                prev_close=1800.0,
                open=1980.0,
                high=1980.0,
                low=1975.0,
                close=1980.0,
                limit_price=1980.0,
                volume=800000.0,
                amount=1584000000.0,
                turnover_rate=6.5,
                volume_ratio=2.0,
                vwap_slope_morning=0.01,
                max_drawdown_morning=-0.002,
                afternoon_strength=0.001,
                quality_score=88.0,
                confidence=0.88
            ),
        ]
        
        # 使用增强筛选器筛选优质股票
        qualified_stocks = self.selector.select_qualified_stocks(
            candidates=mock_stocks,
            min_quality_score=70.0,
            min_confidence=0.5,
            max_open_times=2,
            min_seal_ratio=0.03,
            prefer_first_board=True,
            prefer_sector_leader=True
        )
        
        for stock in qualified_stocks:
            logger.info(
                f"✓ {stock.symbol} {stock.name} - "
                f"连板:{stock.consecutive_days}天, "
                f"封单:{stock.seal_ratio:.1%}, "
                f"质量:{stock.quality_score}分, "
                f"{'首板' if stock.is_first_board else '连板'}, "
                f"{'龙头' if stock.is_sector_leader else ''}"
            )
        
        self.yesterday_limits = qualified_stocks
        logger.info(f"共加载 {len(qualified_stocks)} 只优选股票")
        logger.info("=" * 60)
        return qualified_stocks
    
    async def fetch_auction_data(self, symbol: str) -> Optional[AuctionSnapshot]:
        """
        抓取实时竞价数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            竞价快照
        """
        try:
            # TODO: 接入真实数据源
            # 可选方案:
            # 1. 同花顺Level2接口
            # 2. 东方财富竞价数据
            # 3. 券商API
            # 4. akshare竞价数据
            
            # 模拟数据(实际使用时替换)
            now = datetime.now()
            snapshot = AuctionSnapshot(
                timestamp=now.strftime("%H:%M:%S"),
                symbol=symbol,
                price=10.0 + np.random.uniform(-0.5, 0.5),
                change_pct=np.random.uniform(0, 10),
                bid_volume=int(np.random.uniform(1000000, 5000000)),
                ask_volume=int(np.random.uniform(500000, 2000000)),
                bid_orders=int(np.random.uniform(100, 500)),
                ask_orders=int(np.random.uniform(50, 200)),
                total_volume=int(np.random.uniform(1500000, 7000000)),
                bid_prices=[10.0 + i * 0.01 for i in range(5)],
                ask_prices=[10.0 + i * 0.01 for i in range(5, 10)],
                large_buy_orders=int(np.random.uniform(5, 30)),
                large_sell_orders=int(np.random.uniform(2, 15))
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"抓取 {symbol} 竞价数据失败: {e}")
            return None
    
    def get_current_phase(self) -> AuctionPhase:
        """获取当前竞价阶段"""
        now = datetime.now().time()
        
        if now < self.start_time:
            return AuctionPhase.BEFORE
        elif now < self.critical_time:
            return AuctionPhase.PHASE1
        elif now < self.end_time:
            return AuctionPhase.PHASE2
        elif now < dt_time(9, 30, 0):
            return AuctionPhase.PHASE3
        else:
            return AuctionPhase.AFTER
    
    def analyze_auction_strength(
        self, 
        snapshots: List[AuctionSnapshot]
    ) -> Dict[str, Any]:
        """
        分析集合竞价强度
        
        关键指标:
        1. 竞价涨幅稳定性
        2. 9:20前后价格变化
        3. 买卖盘对比
        4. 大单参与度
        5. 价格趋势
        """
        if not snapshots:
            return {"strength": 0, "confidence": 0}
        
        prices = [s.price for s in snapshots]
        changes = [s.change_pct for s in snapshots]
        bid_volumes = [s.bid_volume for s in snapshots]
        ask_volumes = [s.ask_volume for s in snapshots]
        
        # 1. 价格稳定性
        price_volatility = np.std(prices) / np.mean(prices) if prices else 1
        stability_score = max(0, 100 - price_volatility * 1000)
        
        # 2. 涨幅趋势
        avg_change = np.mean(changes) if changes else 0
        change_trend = np.polyfit(range(len(changes)), changes, 1)[0] if len(changes) > 2 else 0
        
        # 3. 买卖力量对比
        total_bid = sum(bid_volumes)
        total_ask = sum(ask_volumes)
        bid_ask_ratio = total_bid / (total_ask + 1)
        
        # 4. 大单占比
        large_buy = sum(s.large_buy_orders for s in snapshots)
        large_sell = sum(s.large_sell_orders for s in snapshots)
        large_ratio = large_buy / (large_buy + large_sell + 1)
        
        # 综合强度评分
        strength = (
            stability_score * 0.25 +
            min(avg_change * 10, 100) * 0.35 +
            min(bid_ask_ratio * 30, 100) * 0.25 +
            large_ratio * 100 * 0.15
        )
        
        return {
            "strength": min(strength, 100),
            "confidence": 0.85,
            "avg_change": avg_change,
            "stability": stability_score,
            "bid_ask_ratio": bid_ask_ratio,
            "large_ratio": large_ratio,
            "trend": "上涨" if change_trend > 0 else "下跌"
        }
    
    async def monitor_loop(self):
        """监控主循环"""
        logger.info("=" * 60)
        logger.info("🚀 开始9:15-9:26集合竞价实时监控")
        logger.info("=" * 60)
        
        self.is_running = True
        
        while self.is_running:
            now = datetime.now()
            phase = self.get_current_phase()
            
            # 仅在竞价时段内监控
            if phase in [AuctionPhase.PHASE1, AuctionPhase.PHASE2]:
                logger.info(f"\n⏰ {now.strftime('%H:%M:%S')} - {phase.value}")
                
                # 并行抓取所有股票的竞价数据
                tasks = [
                    self.fetch_auction_data(stock.symbol) 
                    for stock in self.yesterday_limits
                ]
                snapshots = await asyncio.gather(*tasks)
                
                # 保存数据
                for stock, snapshot in zip(self.yesterday_limits, snapshots):
                    if snapshot:
                        if stock.symbol not in self.auction_data:
                            self.auction_data[stock.symbol] = []
                        self.auction_data[stock.symbol].append(snapshot)
                        
                        # 分析当前强度
                        analysis = self.analyze_auction_strength(
                            self.auction_data[stock.symbol]
                        )
                        
                        logger.info(
                            f"  📊 {stock.symbol} {stock.name}: "
                            f"价格 {snapshot.price:.2f} "
                            f"涨幅 {snapshot.change_pct:.2f}% "
                            f"强度 {analysis['strength']:.1f} "
                            f"买卖比 {analysis['bid_ask_ratio']:.2f}"
                        )
                
                # 9:20关键时刻特别提示
                if phase == AuctionPhase.PHASE2:
                    logger.warning("⚠️  已进入9:20-9:25不可撤单阶段!")
                
                await asyncio.sleep(self.refresh_interval)
            
            elif phase == AuctionPhase.AFTER:
                logger.info("✅ 集合竞价结束,准备生成分析报告...")
                break
            
            else:
                # 等待竞价开始
                wait_seconds = (
                    datetime.combine(now.date(), self.start_time) - 
                    datetime.combine(now.date(), now.time())
                ).seconds
                if wait_seconds > 0 and wait_seconds < 3600:
                    logger.info(f"⏳ 等待竞价开始,剩余 {wait_seconds} 秒...")
                    await asyncio.sleep(min(60, wait_seconds))
                else:
                    await asyncio.sleep(60)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        生成竞价分析报告
        
        Returns:
            包含所有股票竞价分析的报告
        """
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "stocks_analyzed": len(self.yesterday_limits),
            "stocks": []
        }
        
        for stock in self.yesterday_limits:
            snapshots = self.auction_data.get(stock.symbol, [])
            if not snapshots:
                continue
            
            analysis = self.analyze_auction_strength(snapshots)
            final_snapshot = snapshots[-1]
            
            stock_report = {
                "symbol": stock.symbol,
                "name": stock.name,
                "yesterday_info": {
                    "consecutive_days": stock.consecutive_days,
                    "seal_ratio": stock.seal_ratio,
                    "is_leader": stock.is_leader,
                    "quality_score": stock.quality_score
                },
                "auction_info": {
                    "final_price": final_snapshot.price,
                    "final_change": final_snapshot.change_pct,
                    "strength": analysis["strength"],
                    "stability": analysis["stability"],
                    "bid_ask_ratio": analysis["bid_ask_ratio"],
                    "large_ratio": analysis["large_ratio"],
                    "snapshots_count": len(snapshots)
                }
            }
            
            report["stocks"].append(stock_report)
        
        # 按竞价强度排序
        report["stocks"].sort(
            key=lambda x: x["auction_info"]["strength"], 
            reverse=True
        )
        
        return report
    
    async def start(self):
        """启动监控系统"""
        # 1. 加载昨日涨停股
        self.load_yesterday_limit_up_stocks()
        
        if not self.yesterday_limits:
            logger.warning("未找到符合条件的昨日涨停股,监控取消")
            return None
        
        # 2. 开始监控
        await self.monitor_loop()
        
        # 3. 生成报告
        report = self.generate_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("📋 集合竞价分析报告")
        logger.info("=" * 60)
        for i, stock_data in enumerate(report["stocks"][:10], 1):
            logger.info(
                f"{i}. {stock_data['symbol']} {stock_data['name']}: "
                f"强度 {stock_data['auction_info']['strength']:.1f}, "
                f"涨幅 {stock_data['auction_info']['final_change']:.2f}%, "
                f"买卖比 {stock_data['auction_info']['bid_ask_ratio']:.2f}"
            )
        logger.info("=" * 60)
        
        return report


if __name__ == "__main__":
    monitor = AuctionMonitor(refresh_interval=3)
    
    # 运行监控
    report = asyncio.run(monitor.start())
    
    if report:
        # 保存报告
        import json
        report_file = f"auction_report_{datetime.now():%Y%m%d}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {report_file}")
