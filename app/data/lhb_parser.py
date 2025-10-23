"""
麒麟量化系统 - 龙虎榜数据解析器
解析龙虎榜数据，识别知名游资席位和资金动向

核心功能：
1. 龙虎榜数据获取与解析
2. 知名游资席位识别
3. 席位交易风格分析
4. 游资联动分析
5. 历史成功率统计
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SeatType(Enum):
    """席位类型"""
    HOT_MONEY = "顶级游资"
    INSTITUTION = "机构专用"
    UNKNOWN = "未知席位"


class TradingStyle(Enum):
    """交易风格"""
    AGGRESSIVE = "激进打板"      # 喜欢首板、二板
    STEADY = "稳健接力"          # 喜欢二板以上
    SWING = "波段持有"           # 持有多日
    FAST_IN_OUT = "快进快出"     # 当日买卖


@dataclass
class SeatInfo:
    """席位信息"""
    seat_name: str              # 席位名称
    seat_type: SeatType         # 席位类型
    trading_style: TradingStyle # 交易风格
    win_rate: float            # 历史胜率
    avg_profit: float          # 平均盈利(%)
    famous_level: int          # 知名度等级(1-5)
    activity_score: float      # 活跃度(0-1)
    
    # 特征标签
    prefer_board: str = ""     # 偏好板型(首板/二板/高位板)
    prefer_sector: List[str] = field(default_factory=list)  # 偏好板块
    
    # 历史统计
    total_trades: int = 0      # 总交易次数
    recent_win_rate: float = 0  # 近期胜率
    

@dataclass
class LHBRecord:
    """龙虎榜单条记录"""
    date: str                   # 日期
    symbol: str                 # 股票代码
    stock_name: str            # 股票名称
    close_price: float         # 收盘价
    change_pct: float          # 涨跌幅
    turnover: float            # 成交额(亿)
    
    # 上榜原因
    reason: str                # 上榜原因
    is_limit_up: bool          # 是否涨停
    
    # 买方席位
    buy_seats: List[Dict]      # 买方前5席位
    buy_total: float           # 买入总额(万)
    
    # 卖方席位
    sell_seats: List[Dict]     # 卖方前5席位
    sell_total: float          # 卖出总额(万)
    
    # 净买入
    net_buy: float             # 净买入(万)
    

class FamousSeatDatabase:
    """知名席位数据库"""
    
    def __init__(self):
        self.seats: Dict[str, SeatInfo] = {}
        self._load_famous_seats()
    
    def _load_famous_seats(self):
        """加载知名游资席位库"""
        # 顶级游资（实战中需要持续更新）
        famous_seats = {
            # 超一线游资
            "东方财富证券拉萨东环路第二": SeatInfo(
                seat_name="东方财富证券拉萨东环路第二",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.AGGRESSIVE,
                win_rate=0.65,
                avg_profit=25.0,
                famous_level=5,
                activity_score=0.9,
                prefer_board="首板+二板",
                prefer_sector=["科技", "新能源", "医药"]
            ),
            "湘财证券长沙韶山中路": SeatInfo(
                seat_name="湘财证券长沙韶山中路",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.AGGRESSIVE,
                win_rate=0.62,
                avg_profit=22.0,
                famous_level=5,
                activity_score=0.85,
                prefer_board="首板+二板",
                prefer_sector=["题材股", "科技"]
            ),
            "华泰证券深圳益田路荣超商务中心": SeatInfo(
                seat_name="华泰证券深圳益田路荣超商务中心",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.STEADY,
                win_rate=0.68,
                avg_profit=30.0,
                famous_level=5,
                activity_score=0.8,
                prefer_board="二板+三板",
                prefer_sector=["科技", "消费"]
            ),
            "申万宏源深圳金田路": SeatInfo(
                seat_name="申万宏源深圳金田路",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.AGGRESSIVE,
                win_rate=0.60,
                avg_profit=20.0,
                famous_level=4,
                activity_score=0.75,
                prefer_board="首板",
                prefer_sector=["题材股"]
            ),
            "国泰君安成都北一环路": SeatInfo(
                seat_name="国泰君安成都北一环路",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.STEADY,
                win_rate=0.64,
                avg_profit=28.0,
                famous_level=4,
                activity_score=0.7,
                prefer_board="二板+高位板",
                prefer_sector=["科技", "军工"]
            ),
            
            # 一线游资
            "中信证券上海分公司": SeatInfo(
                seat_name="中信证券上海分公司",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.SWING,
                win_rate=0.58,
                avg_profit=18.0,
                famous_level=3,
                activity_score=0.65,
                prefer_board="所有",
                prefer_sector=["白马股", "蓝筹"]
            ),
            "中信建投北京安立路": SeatInfo(
                seat_name="中信建投北京安立路",
                seat_type=SeatType.HOT_MONEY,
                trading_style=TradingStyle.FAST_IN_OUT,
                win_rate=0.55,
                avg_profit=15.0,
                famous_level=3,
                activity_score=0.6,
                prefer_board="首板",
                prefer_sector=["题材股"]
            ),
        }
        
        # 机构专用席位
        institution_seats = {
            "机构专用": SeatInfo(
                seat_name="机构专用",
                seat_type=SeatType.INSTITUTION,
                trading_style=TradingStyle.SWING,
                win_rate=0.52,
                avg_profit=12.0,
                famous_level=2,
                activity_score=0.4,
                prefer_board="所有"
            ),
        }
        
        self.seats.update(famous_seats)
        self.seats.update(institution_seats)
        
        logger.info(f"已加载 {len(self.seats)} 个知名席位")
    
    def identify_seat(self, seat_name: str) -> Optional[SeatInfo]:
        """识别席位"""
        # 精确匹配
        if seat_name in self.seats:
            return self.seats[seat_name]
        
        # 模糊匹配（关键词）
        for known_seat, info in self.seats.items():
            # 提取关键词
            keywords = known_seat.split("证券")
            if len(keywords) > 1:
                key = keywords[1].split("路")[0] if "路" in keywords[1] else keywords[1][:4]
                if key in seat_name:
                    return info
        
        # 机构专用
        if "机构专用" in seat_name:
            return self.seats.get("机构专用")
        
        return None
    
    def get_seat_rank(self, seat_name: str) -> int:
        """获取席位等级（1-5，5为最高）"""
        seat_info = self.identify_seat(seat_name)
        return seat_info.famous_level if seat_info else 0


class LHBParser:
    """龙虎榜解析器"""
    
    def __init__(self, data_source: str = "akshare"):
        """
        初始化解析器
        
        Args:
            data_source: 数据源（akshare/tushare/自定义）
        """
        self.data_source = data_source
        self.seat_db = FamousSeatDatabase()
        
        # 缓存
        self.cache: Dict[str, List[LHBRecord]] = {}
        self.cache_expiry = timedelta(hours=1)
        
        logger.info(f"龙虎榜解析器初始化完成，数据源: {data_source}")
    
    def fetch_lhb_data(self, date: str) -> List[LHBRecord]:
        """
        获取指定日期的龙虎榜数据
        
        Args:
            date: 日期字符串 YYYY-MM-DD
            
        Returns:
            龙虎榜记录列表
        """
        # 检查缓存
        if date in self.cache:
            logger.info(f"从缓存读取 {date} 龙虎榜数据")
            return self.cache[date]
        
        # 实际应用中从数据源获取
        if self.data_source == "akshare":
            records = self._fetch_from_akshare(date)
        elif self.data_source == "tushare":
            records = self._fetch_from_tushare(date)
        else:
            # 模拟数据
            records = self._generate_mock_data(date)
        
        # 缓存
        self.cache[date] = records
        
        logger.info(f"获取 {date} 龙虎榜数据: {len(records)} 条")
        return records
    
    def _fetch_from_akshare(self, date: str) -> List[LHBRecord]:
        """从AKShare获取数据"""
        try:
            import akshare as ak
            
            # 获取龙虎榜数据
            df = ak.stock_lhb_detail_em(date=date.replace("-", ""))
            
            records = []
            for _, row in df.iterrows():
                # 解析买卖席位
                buy_seats = self._parse_seats(row, "buy")
                sell_seats = self._parse_seats(row, "sell")
                
                record = LHBRecord(
                    date=date,
                    symbol=row.get('代码', ''),
                    stock_name=row.get('名称', ''),
                    close_price=float(row.get('收盘价', 0)),
                    change_pct=float(row.get('涨跌幅', 0)),
                    turnover=float(row.get('成交额', 0)) / 1e8,
                    reason=row.get('上榜原因', ''),
                    is_limit_up='涨停' in row.get('上榜原因', ''),
                    buy_seats=buy_seats,
                    buy_total=sum(s['amount'] for s in buy_seats),
                    sell_seats=sell_seats,
                    sell_total=sum(s['amount'] for s in sell_seats),
                    net_buy=sum(s['amount'] for s in buy_seats) - sum(s['amount'] for s in sell_seats)
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"AKShare获取龙虎榜失败: {e}")
            return []
    
    def _fetch_from_tushare(self, date: str) -> List[LHBRecord]:
        """从Tushare获取数据"""
        # 类似实现
        logger.warning("Tushare接口待实现")
        return []
    
    def _generate_mock_data(self, date: str) -> List[LHBRecord]:
        """生成模拟数据"""
        mock_records = [
            LHBRecord(
                date=date,
                symbol="000001",
                stock_name="平安银行",
                close_price=11.55,
                change_pct=10.0,
                turnover=55.0,
                reason="涨停",
                is_limit_up=True,
                buy_seats=[
                    {"seat": "东方财富证券拉萨东环路第二", "amount": 5000},
                    {"seat": "湘财证券长沙韶山中路", "amount": 3000},
                    {"seat": "华泰证券深圳益田路荣超商务中心", "amount": 2500},
                ],
                buy_total=10500,
                sell_seats=[
                    {"seat": "机构专用", "amount": 2000},
                    {"seat": "散户席位", "amount": 1500},
                ],
                sell_total=3500,
                net_buy=7000
            ),
        ]
        return mock_records
    
    def _parse_seats(self, row: pd.Series, direction: str) -> List[Dict]:
        """解析席位数据"""
        seats = []
        prefix = "买" if direction == "buy" else "卖"
        
        for i in range(1, 6):
            seat_col = f"{prefix}{i}"
            amount_col = f"{prefix}{i}金额"
            
            if seat_col in row and amount_col in row:
                seat_name = row[seat_col]
                amount = float(row[amount_col]) / 10000  # 转万元
                
                if pd.notna(seat_name) and amount > 0:
                    seats.append({
                        "seat": seat_name,
                        "amount": amount
                    })
        
        return seats
    
    def analyze_stock_seats(self, symbol: str, date: str) -> Dict[str, Any]:
        """
        分析指定股票的席位情况
        
        Args:
            symbol: 股票代码
            date: 日期
            
        Returns:
            席位分析结果
        """
        records = self.fetch_lhb_data(date)
        
        # 找到该股票记录
        stock_record = None
        for record in records:
            if record.symbol == symbol:
                stock_record = record
                break
        
        if not stock_record:
            return {
                "found": False,
                "message": "该股票未上榜"
            }
        
        # 分析买方席位
        buy_analysis = self._analyze_seats(stock_record.buy_seats, "buy")
        
        # 分析卖方席位
        sell_analysis = self._analyze_seats(stock_record.sell_seats, "sell")
        
        # 综合判断
        overall_signal = self._generate_overall_signal(
            buy_analysis, sell_analysis, stock_record
        )
        
        return {
            "found": True,
            "stock_info": {
                "symbol": stock_record.symbol,
                "name": stock_record.stock_name,
                "close": stock_record.close_price,
                "change_pct": stock_record.change_pct,
                "reason": stock_record.reason
            },
            "buy_analysis": buy_analysis,
            "sell_analysis": sell_analysis,
            "net_buy": stock_record.net_buy,
            "overall_signal": overall_signal,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_seats(self, seats: List[Dict], direction: str) -> Dict:
        """分析席位列表"""
        famous_seats = []
        institution_count = 0
        total_amount = 0
        
        for seat_info in seats:
            seat_name = seat_info['seat']
            amount = seat_info['amount']
            total_amount += amount
            
            # 识别席位
            identified = self.seat_db.identify_seat(seat_name)
            
            if identified:
                if identified.seat_type == SeatType.HOT_MONEY:
                    famous_seats.append({
                        "name": seat_name,
                        "amount": amount,
                        "level": identified.famous_level,
                        "style": identified.trading_style.value,
                        "win_rate": identified.win_rate
                    })
                elif identified.seat_type == SeatType.INSTITUTION:
                    institution_count += 1
        
        # 评分
        score = 0
        if famous_seats:
            # 根据知名度和金额加权
            for seat in famous_seats:
                score += seat['level'] * 10 + (seat['amount'] / 1000) * 5
        
        return {
            "famous_seats": famous_seats,
            "institution_count": institution_count,
            "total_amount": total_amount,
            "famous_count": len(famous_seats),
            "score": min(score, 100),
            "direction": direction
        }
    
    def _generate_overall_signal(
        self,
        buy_analysis: Dict,
        sell_analysis: Dict,
        record: LHBRecord
    ) -> Dict:
        """生成综合信号"""
        signal_strength = 0
        reasons = []
        
        # 1. 买方游资实力
        if buy_analysis['famous_count'] >= 3:
            signal_strength += 30
            reasons.append(f"买方有{buy_analysis['famous_count']}个知名游资")
        elif buy_analysis['famous_count'] >= 1:
            signal_strength += 15
            reasons.append(f"买方有游资介入")
        
        # 2. 净买入
        if record.net_buy > 5000:  # 5000万以上
            signal_strength += 25
            reasons.append(f"净买入{record.net_buy/10000:.2f}亿")
        elif record.net_buy > 0:
            signal_strength += 10
        
        # 3. 卖方情况
        if sell_analysis['famous_count'] > 0:
            signal_strength -= 15
            reasons.append(f"卖方有{sell_analysis['famous_count']}个游资离场")
        
        # 4. 涨停板加成
        if record.is_limit_up:
            signal_strength += 10
            reasons.append("涨停上榜")
        
        # 判断信号
        if signal_strength >= 60:
            signal = "强烈看多"
        elif signal_strength >= 40:
            signal = "看多"
        elif signal_strength >= 20:
            signal = "中性偏多"
        elif signal_strength > 0:
            signal = "中性"
        else:
            signal = "谨慎"
        
        return {
            "signal": signal,
            "strength": signal_strength,
            "reasons": reasons
        }
    
    def get_seat_activity(self, seat_name: str, days: int = 30) -> Dict:
        """
        获取席位近期活跃度
        
        Args:
            seat_name: 席位名称
            days: 统计天数
            
        Returns:
            活跃度统计
        """
        # 实战中需要查询历史数据库
        # 这里返回模拟数据
        
        identified = self.seat_db.identify_seat(seat_name)
        if not identified:
            return {"found": False}
        
        return {
            "found": True,
            "seat_name": seat_name,
            "activity_score": identified.activity_score,
            "win_rate": identified.win_rate,
            "avg_profit": identified.avg_profit,
            "trading_style": identified.trading_style.value,
            "prefer_board": identified.prefer_board
        }


# 使用示例
if __name__ == "__main__":
    # 创建解析器
    parser = LHBParser(data_source="mock")
    
    # 获取龙虎榜数据
    date = "2024-12-20"
    records = parser.fetch_lhb_data(date)
    
    print(f"获取到 {len(records)} 条龙虎榜数据")
    
    # 分析单只股票
    if records:
        symbol = records[0].symbol
        analysis = parser.analyze_stock_seats(symbol, date)
        
        print("\n" + "=" * 60)
        print("龙虎榜分析结果")
        print("=" * 60)
        print(f"股票: {analysis['stock_info']['name']} ({analysis['stock_info']['symbol']})")
        print(f"涨跌幅: {analysis['stock_info']['change_pct']:.2f}%")
        print(f"\n买方分析:")
        print(f"  知名游资: {analysis['buy_analysis']['famous_count']}个")
        print(f"  买入总额: {analysis['buy_analysis']['total_amount']/10000:.2f}亿")
        print(f"  评分: {analysis['buy_analysis']['score']:.0f}分")
        
        if analysis['buy_analysis']['famous_seats']:
            print(f"\n  知名席位:")
            for seat in analysis['buy_analysis']['famous_seats']:
                print(f"    - {seat['name']}: {seat['amount']/10000:.2f}亿 (等级{seat['level']})")
        
        print(f"\n综合信号: {analysis['overall_signal']['signal']}")
        print(f"信号强度: {analysis['overall_signal']['strength']}分")
        print(f"理由: {'; '.join(analysis['overall_signal']['reasons'])}")
