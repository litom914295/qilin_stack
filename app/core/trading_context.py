"""
麒麟量化系统 - 统一交易上下文管理器
管理D日历史数据、T+1日盘前数据和实时数据
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class StockDayData:
    """单只股票的日内数据"""
    symbol: str
    date: str
    
    # 基础行情
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    volume: float = 0
    turnover: float = 0
    turnover_rate: float = 0
    
    # 涨停相关
    is_limit_up: bool = False
    limit_up_time: Optional[str] = None  # 涨停时间
    seal_amount: float = 0  # 封单金额
    seal_ratio: float = 0  # 封单比例
    open_times: int = 0  # 开板次数
    limit_type: str = ""  # 涨停类型：一字板/秒板/烂板回封
    consecutive_limit_days: int = 0  # 连板天数
    
    # 资金流向
    main_net_inflow: float = 0  # 主力净流入
    super_large_net: float = 0  # 超大单净额
    large_net: float = 0  # 大单净额
    medium_net: float = 0  # 中单净额
    small_net: float = 0  # 小单净额
    
    # 板块信息
    sector: str = ""
    sector_rank: int = 0  # 板块内涨幅排名
    sector_change: float = 0  # 板块涨幅
    
    # 市场地位
    market_rank: int = 0  # 全市场涨幅排名
    is_leader: bool = False  # 是否龙头
    

@dataclass
class AuctionData:
    """集合竞价数据"""
    symbol: str
    date: str
    time: str
    
    # 价格数据
    price: float = 0
    change_pct: float = 0  # 涨跌幅
    
    # 成交数据
    volume: float = 0
    turnover: float = 0
    volume_ratio: float = 0  # 相对昨日同时段量比
    
    # 委托数据
    bid_volume: float = 0  # 买盘委托量
    ask_volume: float = 0  # 卖盘委托量
    bid_ask_ratio: float = 0  # 委买委卖比
    
    # 时间序列（每30秒一个点）
    price_series: List[float] = field(default_factory=list)
    volume_series: List[float] = field(default_factory=list)
    
    # 分析指标
    is_stable: bool = False  # 价格是否稳定
    trend: str = ""  # 趋势：上升/下降/震荡
    strength: float = 0  # 竞价强度评分


@dataclass 
class MarketSentiment:
    """市场情绪数据"""
    date: str
    
    # 涨跌统计
    total_stocks: int = 0
    up_count: int = 0
    down_count: int = 0
    limit_up_count: int = 0
    limit_down_count: int = 0
    
    # 涨停板统计
    natural_limit_up: int = 0  # 自然涨停（非一字）
    second_board_count: int = 0  # 2连板数
    third_board_count: int = 0  # 3连板数
    high_board_count: int = 0  # 高位板数（4板及以上）
    
    # 板块统计
    hot_sectors: List[Dict] = field(default_factory=list)  # 热门板块
    sector_money_flow: Dict[str, float] = field(default_factory=dict)  # 板块资金流向
    
    # 情绪指标
    sentiment_score: float = 50  # 情绪分数 0-100
    money_effect: float = 0  # 赚钱效应
    loss_effect: float = 0  # 亏钱效应
    
    # 龙虎榜
    lhb_net_buy: float = 0  # 龙虎榜净买入
    famous_seats_buy: List[str] = field(default_factory=list)  # 知名游资买入
    

@dataclass
class NewsData:
    """新闻数据"""
    symbol: str
    timestamp: datetime
    title: str
    content: str = ""
    source: str = ""
    sentiment: float = 0  # 情绪分：-1到1
    importance: int = 0  # 重要性：1-5
    keywords: List[str] = field(default_factory=list)
    

class TradingContext:
    """
    统一的交易上下文
    管理所有时间线的数据：D日历史、T+1日盘前、实时数据
    """
    
    def __init__(self, symbol: str = "TEST", current_time: Optional[datetime] = None):
        """
        初始化交易上下文
        
        Args:
            symbol: 股票代码
            current_time: 当前时间（T+1日的某个时刻）
        """
        self.symbol = symbol
        self.current_time = current_time or datetime.now()
        self.trade_date = self.current_time.strftime('%Y-%m-%d')
        
        # 计算关键时间点
        self.t1_date = self.trade_date  # T+1日（今天）
        self.d_date = self._get_previous_trade_date(self.current_time)  # D日（上一个交易日）
        self.d_minus_1_date = self._get_previous_trade_date(
            datetime.strptime(self.d_date, '%Y-%m-%d')
        )  # D-1日
        
        # D日数据（昨天收盘后的完整数据）
        self.d_day_data: Optional[StockDayData] = None
        self.d_day_market: Optional[MarketSentiment] = None
        self.d_day_limit_up_pool: List[StockDayData] = []  # D日涨停池
        
        # D-1日数据（前天的数据，用于对比）
        self.d_minus_1_data: Optional[StockDayData] = None
        
        # T+1日盘前数据
        self.t1_auction_data: Optional[AuctionData] = None  # 竞价数据
        self.t1_pre_news: List[NewsData] = []  # 盘前新闻
        
        # 实时数据
        self.realtime_quote: Dict[str, Any] = {}
        self.realtime_level2: Dict[str, Any] = {}  # Level2数据
        
        # 衍生数据
        self.technical_indicators: Dict[str, float] = {}
        self.fundamental_data: Dict[str, Any] = {}
        
        # 数据完整性标记
        self.data_completeness: Dict[str, bool] = {
            'd_day_data': False,
            'd_day_market': False,
            't1_auction': False,
            'realtime': False
        }
        
        logger.info(f"初始化交易上下文 - 股票:{symbol}, D日:{self.d_date}, T+1日:{self.t1_date}")

    def create_context(self, symbol: str, d_day_historical: pd.DataFrame, t1_premarket: pd.DataFrame) -> Dict[str, Any]:
        """创建统一上下文字典（用于测试）"""
        return {
            'symbol': symbol,
            'd_day_data': d_day_historical,
            't1_data': t1_premarket,
        }

    def validate_data(self, data: pd.DataFrame, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """简单数据校验：非空且包含必需字段"""
        errors: List[str] = []
        if data is None or data.empty:
            errors.append("数据为空")
        missing = [f for f in required_fields if f not in data.columns]
        if missing:
            errors.append(f"缺失必需字段: {', '.join(missing)}")
        return (len(errors) == 0), errors
    
    def _get_previous_trade_date(self, date: datetime) -> str:
        """获取前一个交易日（简化版，实际需要交易日历）"""
        # 这里简化处理，实际需要使用交易日历
        if date.weekday() == 0:  # 周一
            prev_date = date - timedelta(days=3)
        elif date.weekday() == 6:  # 周日
            prev_date = date - timedelta(days=2)
        else:
            prev_date = date - timedelta(days=1)
        return prev_date.strftime('%Y-%m-%d')
    
    def load_d_day_data(self) -> bool:
        """加载D日完整数据"""
        try:
            # 加载个股D日数据
            self.d_day_data = self._load_stock_day_data(self.symbol, self.d_date)
            
            # 加载D日市场数据
            self.d_day_market = self._load_market_sentiment(self.d_date)
            
            # 加载D日涨停池
            self.d_day_limit_up_pool = self._load_limit_up_pool(self.d_date)
            
            # 加载D-1日数据用于对比
            self.d_minus_1_data = self._load_stock_day_data(self.symbol, self.d_minus_1_date)
            
            self.data_completeness['d_day_data'] = True
            self.data_completeness['d_day_market'] = True
            
            logger.info(f"D日数据加载完成 - 涨停池:{len(self.d_day_limit_up_pool)}只")
            return True
            
        except Exception as e:
            logger.error(f"加载D日数据失败: {e}")
            return False
    
    def load_t1_auction_data(self) -> bool:
        """加载T+1日集合竞价数据"""
        try:
            # 加载竞价数据
            self.t1_auction_data = self._load_auction_data(self.symbol, self.t1_date)
            
            # 加载盘前新闻
            self.t1_pre_news = self._load_pre_market_news(self.symbol, self.t1_date)
            
            self.data_completeness['t1_auction'] = True
            
            logger.info(f"T+1竞价数据加载完成 - 涨幅:{self.t1_auction_data.change_pct:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"加载T+1竞价数据失败: {e}")
            return False
    
    def load_realtime_data(self) -> bool:
        """加载实时数据"""
        try:
            # 加载实时行情
            self.realtime_quote = self._load_realtime_quote(self.symbol)
            
            # 加载Level2数据
            self.realtime_level2 = self._load_level2_data(self.symbol)
            
            self.data_completeness['realtime'] = True
            
            return True
            
        except Exception as e:
            logger.error(f"加载实时数据失败: {e}")
            return False
    
    def _load_stock_day_data(self, symbol: str, date: str) -> StockDayData:
        """加载个股日数据（实际实现需要对接数据源）"""
        # 这里是模拟数据，实际需要从数据库或API获取
        data = StockDayData(
            symbol=symbol,
            date=date,
            open=10.5,
            high=11.55,
            low=10.3,
            close=11.55,  # 涨停
            volume=5000000,
            turnover=55000000,
            turnover_rate=8.5,
            is_limit_up=True,
            limit_up_time="10:30",
            seal_amount=20000000,
            seal_ratio=0.08,
            open_times=1,
            limit_type="烂板回封",
            consecutive_limit_days=2,
            main_net_inflow=15000000,
            sector="新能源",
            sector_rank=3,
            sector_change=3.5,
            market_rank=15
        )
        return data
    
    def _load_market_sentiment(self, date: str) -> MarketSentiment:
        """加载市场情绪数据"""
        sentiment = MarketSentiment(
            date=date,
            total_stocks=5000,
            up_count=3200,
            down_count=1700,
            limit_up_count=145,
            limit_down_count=8,
            natural_limit_up=89,
            second_board_count=35,
            third_board_count=12,
            high_board_count=5,
            hot_sectors=[
                {"name": "新能源", "change": 4.5, "leader": "300750"},
                {"name": "半导体", "change": 3.8, "leader": "603986"},
                {"name": "军工", "change": 3.2, "leader": "600760"}
            ],
            sentiment_score=72,  # 情绪偏热
            money_effect=0.65,
            loss_effect=0.15,
            lhb_net_buy=580000000
        )
        return sentiment
    
    def _load_limit_up_pool(self, date: str) -> List[StockDayData]:
        """加载涨停池数据"""
        # 模拟涨停池，实际需要从数据源加载
        pool = []
        
        # 模拟几只涨停股
        symbols = ["300750", "002415", "603986", "000333", "600760"]
        for sym in symbols:
            stock = self._load_stock_day_data(sym, date)
            if stock.is_limit_up:
                pool.append(stock)
        
        return pool
    
    def _load_auction_data(self, symbol: str, date: str) -> AuctionData:
        """加载集合竞价数据"""
        # 模拟竞价数据
        auction = AuctionData(
            symbol=symbol,
            date=date,
            time="09:25",
            price=11.8,
            change_pct=7.2,
            volume=1500000,
            turnover=17700000,
            volume_ratio=2.5,
            bid_volume=2000000,
            ask_volume=800000,
            bid_ask_ratio=2.5,
            price_series=[11.5, 11.6, 11.7, 11.75, 11.8],  # 09:20-09:25每分钟价格
            volume_series=[200000, 300000, 400000, 350000, 250000],
            is_stable=True,
            trend="上升",
            strength=85
        )
        return auction
    
    def _load_pre_market_news(self, symbol: str, date: str) -> List[NewsData]:
        """加载盘前新闻"""
        news_list = [
            NewsData(
                symbol=symbol,
                timestamp=datetime.now(),
                title=f"{symbol}获得重大订单，预计增厚全年业绩",
                sentiment=0.8,
                importance=4,
                keywords=["订单", "业绩", "利好"]
            ),
            NewsData(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(hours=1),
                title=f"机构密集调研{symbol}，看好公司发展前景",
                sentiment=0.6,
                importance=3,
                keywords=["机构", "调研", "前景"]
            )
        ]
        return news_list
    
    def _load_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """加载实时行情"""
        return {
            'current': 11.85,
            'change_pct': 7.5,
            'volume': 2000000,
            'bid1': 11.85,
            'ask1': 11.86,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_level2_data(self, symbol: str) -> Dict[str, Any]:
        """加载Level2数据"""
        return {
            'buy_orders': [
                {'price': 11.85, 'volume': 50000, 'count': 25},
                {'price': 11.84, 'volume': 30000, 'count': 18},
            ],
            'sell_orders': [
                {'price': 11.86, 'volume': 10000, 'count': 8},
                {'price': 11.87, 'volume': 15000, 'count': 12},
            ],
            'transaction_list': [],  # 逐笔成交
            'order_queue': {}  # 委托队列
        }
    
    def calculate_technical_indicators(self):
        """计算技术指标"""
        if not self.d_day_data:
            return
        
        # RSI
        self.technical_indicators['rsi'] = 65  # 模拟值
        
        # MACD
        self.technical_indicators['macd'] = 0.15
        
        # 布林带
        self.technical_indicators['bb_upper'] = 12.0
        self.technical_indicators['bb_middle'] = 11.0
        self.technical_indicators['bb_lower'] = 10.0
        
        # 支撑位和压力位
        self.technical_indicators['support'] = 10.5
        self.technical_indicators['resistance'] = 12.0
        
    def get_completeness_report(self) -> Dict[str, Any]:
        """获取数据完整性报告"""
        total_items = len(self.data_completeness)
        completed_items = sum(self.data_completeness.values())
        
        return {
            'completeness_rate': completed_items / total_items,
            'details': self.data_completeness,
            'missing': [k for k, v in self.data_completeness.items() if not v]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于传递给Agent"""
        return {
            'symbol': self.symbol,
            'current_time': self.current_time.isoformat(),
            'd_date': self.d_date,
            't1_date': self.t1_date,
            'd_day_data': self.d_day_data.__dict__ if self.d_day_data else {},
            'd_day_market': self.d_day_market.__dict__ if self.d_day_market else {},
            'd_day_limit_up_pool': [s.__dict__ for s in self.d_day_limit_up_pool],
            't1_auction_data': self.t1_auction_data.__dict__ if self.t1_auction_data else {},
            't1_pre_news': [n.__dict__ for n in self.t1_pre_news],
            'realtime_quote': self.realtime_quote,
            'technical_indicators': self.technical_indicators,
            'data_completeness': self.data_completeness
        }


class ContextManager:
    """上下文管理器，负责批量管理多只股票的上下文"""
    
    def __init__(self, current_time: datetime):
        self.current_time = current_time
        self.contexts: Dict[str, TradingContext] = {}
        
    def create_context(self, symbol: str) -> TradingContext:
        """创建单只股票的上下文"""
        ctx = TradingContext(symbol, self.current_time)
        self.contexts[symbol] = ctx
        return ctx
    
    def load_all_data(self, symbols: List[str]) -> Dict[str, TradingContext]:
        """批量加载所有股票的数据"""
        for symbol in symbols:
            ctx = self.create_context(symbol)
            
            # 加载各类数据
            ctx.load_d_day_data()
            ctx.load_t1_auction_data()
            ctx.load_realtime_data()
            ctx.calculate_technical_indicators()
            
            # 检查完整性
            report = ctx.get_completeness_report()
            if report['completeness_rate'] < 0.8:
                logger.warning(f"{symbol} 数据不完整: {report['missing']}")
        
        return self.contexts
    
    def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览"""
        if not self.contexts:
            return {}
        
        # 使用第一个股票的市场数据（所有股票共享市场数据）
        first_ctx = next(iter(self.contexts.values()))
        if first_ctx.d_day_market:
            return {
                'date': first_ctx.d_date,
                'sentiment_score': first_ctx.d_day_market.sentiment_score,
                'limit_up_count': first_ctx.d_day_market.limit_up_count,
                'money_effect': first_ctx.d_day_market.money_effect,
                'hot_sectors': first_ctx.d_day_market.hot_sectors
            }
        return {}


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建上下文管理器（T+1日早上8:55）
    current_time = datetime(2024, 12, 20, 8, 55, 0)
    manager = ContextManager(current_time)
    
    # 批量加载股票数据
    symbols = ['000001', '000002', '300750']
    contexts = manager.load_all_data(symbols)
    
    # 获取单只股票的完整上下文
    ctx = contexts['000001']
    print(f"股票: {ctx.symbol}")
    print(f"D日涨停: {ctx.d_day_data.is_limit_up}")
    print(f"竞价涨幅: {ctx.t1_auction_data.change_pct:.2f}%")
    print(f"数据完整性: {ctx.get_completeness_report()}")
    
    # 获取市场概览
    market = manager.get_market_overview()
    print(f"市场情绪: {market.get('sentiment_score', 0)}")