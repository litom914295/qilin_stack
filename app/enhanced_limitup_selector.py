"""
麒麟量化系统 - 增强版昨日涨停板筛选器
整合 advanced-ak-pack 的首板识别和分时特征逻辑
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from risk_management.market_timing import is_safe_environment

logger = logging.getLogger(__name__)


@dataclass
class LimitUpStock:
    """涨停股详细信息"""
    symbol: str
    name: str
    date: str
    
    # 涨停基础信息
    limit_up_time: str          # 涨停时间
    open_times: int             # 开板次数
    seal_ratio: float           # 封单强度
    is_one_word: bool          # 是否一字板
    
    # 连板信息
    consecutive_days: int       # 连板天数
    is_first_board: bool       # 是否首板
    prev_limit_up: bool        # 前日是否涨停
    
    # 板块题材
    sector: str                # 所属板块
    themes: List[str]          # 题材概念
    sector_limit_count: int    # 当日板块涨停数
    is_sector_leader: bool     # 是否板块龙头
    
    # 价格信息
    prev_close: float
    open: float
    high: float
    low: float
    close: float
    limit_price: float
    
    # 成交量信息
    volume: float
    amount: float
    turnover_rate: float
    volume_ratio: float
    
    # 分时表现 (当日盘中)
    vwap_slope_morning: Optional[float]      # 早盘VWAP斜率
    max_drawdown_morning: Optional[float]    # 早盘最大回撤
    afternoon_strength: Optional[float]      # 午后强度
    
    # 质量评分
    quality_score: float        # 综合质量分
    confidence: float          # 置信度


class EnhancedLimitUpSelector:
    """增强版涨停板筛选器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def infer_limit_ratio(self, symbol: str) -> float:
        """推断涨跌幅限制"""
        if symbol.startswith(('688', '689')):  # 科创板
            return 0.20
        elif symbol.startswith('3'):           # 创业板(注册制后)
            return 0.20
        elif symbol.startswith(('N', 'C')):    # 新股首日
            return 0.44  # 44% (实际更复杂)
        elif symbol.startswith('ST') or 'ST' in symbol:  # ST股
            return 0.05
        else:
            return 0.10  # 主板
    
    def calculate_limit_price(self, prev_close: float, symbol: str) -> float:
        """计算涨停价"""
        ratio = self.infer_limit_ratio(symbol)
        return round(prev_close * (1 + ratio), 2)
    
    def is_limit_up(
        self, 
        close: float, 
        high: float, 
        prev_close: float, 
        symbol: str,
        threshold: float = 0.999
    ) -> bool:
        """
        判断是否涨停
        
        Args:
            close: 收盘价
            high: 最高价
            prev_close: 前收盘价
            symbol: 股票代码
            threshold: 涨停判断阈值(默认99.9%)
        """
        if not (np.isfinite(prev_close) and prev_close > 0):
            return False
        
        limit_price = self.calculate_limit_price(prev_close, symbol)
        
        # 收盘价或最高价触及涨停价
        return (close >= limit_price * threshold) or (high >= limit_price * threshold)
    
    def detect_first_board(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """
        识别首板(借鉴advanced-ak-pack逻辑)
        
        首板定义: T日涨停 且 T-1日未涨停
        
        Args:
            df: 包含至少3天数据的DataFrame [date, open, high, low, close, volume]
            symbol: 股票代码
            
        Returns:
            添加了首板标记的DataFrame
        """
        df = df.sort_values("date").copy()
        
        if len(df) < 2:
            df["is_first_board"] = False
            return df
        
        # 计算前收盘
        df["prev_close"] = df["close"].shift(1)
        df["pprev_close"] = df["close"].shift(2)
        
        # 判断当日涨停
        df["is_limit_up"] = df.apply(
            lambda row: self.is_limit_up(
                row["close"], row["high"], row["prev_close"], symbol
            ) if np.isfinite(row["prev_close"]) else False,
            axis=1
        )
        
        # 判断前日是否涨停
        df["prev_limit_up"] = df.apply(
            lambda row: self.is_limit_up(
                row["prev_close"], row["prev_close"], row["pprev_close"], symbol
            ) if np.isfinite(row["pprev_close"]) else False,
            axis=1
        )
        
        # 首板 = 当日涨停 且 前日未涨停
        df["is_first_board"] = df["is_limit_up"] & (~df["prev_limit_up"])
        
        return df
    
    def calculate_consecutive_days(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> int:
        """
        计算连板天数
        
        Args:
            df: 历史日线数据
            symbol: 股票代码
            
        Returns:
            连板天数
        """
        df = df.sort_values("date", ascending=False).copy()
        
        consecutive = 0
        prev_close = None
        
        for _, row in df.iterrows():
            if prev_close is None:
                prev_close = row.get("prev_close", 0)
            
            if self.is_limit_up(row["close"], row["high"], prev_close, symbol):
                consecutive += 1
                prev_close = row["prev_close"] if "prev_close" in row else 0
            else:
                break
        
        return consecutive
    
    def extract_intraday_features(
        self, 
        minute_df: pd.DataFrame,
        morning_window: int = 60
    ) -> Dict[str, float]:
        """
        提取分时特征(借鉴advanced-ak-pack)
        
        Args:
            minute_df: 分时数据 [datetime, open, close, high, low, volume, amount]
            morning_window: 早盘窗口(分钟)
            
        Returns:
            分时特征字典
        """
        if minute_df is None or minute_df.empty:
            return {
                "vwap_slope_morning": np.nan,
                "max_drawdown_morning": np.nan,
                "afternoon_strength": np.nan
            }
        
        # 计算VWAP
        minute_df = minute_df.copy()
        minute_df["cum_amount"] = minute_df["amount"].cumsum()
        minute_df["cum_volume"] = minute_df["volume"].cumsum().replace(0, np.nan)
        minute_df["vwap"] = minute_df["cum_amount"] / minute_df["cum_volume"]
        
        # 早盘数据
        morning_data = minute_df.head(morning_window)
        
        # 1. 早盘VWAP斜率
        vwap_slope = np.nan
        if len(morning_data) >= 10:
            y = morning_data["vwap"].dropna().values
            if len(y) >= 5:
                x = np.arange(len(y))
                x_mean, y_mean = x.mean(), y.mean()
                denom = ((x - x_mean) ** 2).sum()
                if denom > 0:
                    vwap_slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        
        # 2. 早盘最大回撤
        max_drawdown = np.nan
        if len(morning_data) >= 1:
            base_open = morning_data["open"].iloc[0]
            if base_open and np.isfinite(base_open) and base_open > 0:
                returns = (morning_data["close"] / base_open) - 1
                cumulative = (returns + 1).cumprod()
                max_drawdown = float((cumulative / cumulative.cummax() - 1).min())
        
        # 3. 午后强度 (13:00之后)
        afternoon_data = minute_df[
            minute_df["datetime"].dt.strftime("%H:%M:%S") >= "13:00:00"
        ]
        
        afternoon_strength = np.nan
        if not afternoon_data.empty and len(afternoon_data) >= 1:
            afternoon_strength = float(
                (afternoon_data["close"].iloc[-1] / afternoon_data["open"].iloc[0]) - 1
            )
        
        return {
            "vwap_slope_morning": vwap_slope,
            "max_drawdown_morning": max_drawdown,
            "afternoon_strength": afternoon_strength
        }
    
    def calculate_quality_score(self, stock: LimitUpStock) -> float:
        """
        计算涨停质量分(0-100)
        
        综合考虑:
        - 封单强度 (25分)
        - 涨停时间 (20分)
        - 开板次数 (20分)
        - 换手率 (15分)
        - 板块热度 (10分)
        - 连板天数 (10分)
        """
        score = 0.0
        
        # 1. 封单强度 (25分)
        if stock.seal_ratio >= 0.15:
            score += 25
        elif stock.seal_ratio >= 0.08:
            score += 20
        elif stock.seal_ratio >= 0.05:
            score += 15
        elif stock.seal_ratio >= 0.03:
            score += 10
        
        # 2. 涨停时间 (20分) - 越早越好
        if stock.limit_up_time:
            time_parts = stock.limit_up_time.split(":")
            if len(time_parts) >= 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                
                if hour == 9 and minute <= 30:
                    score += 20  # 开盘即涨停
                elif hour == 9:
                    score += 18
                elif hour == 10:
                    score += 15
                elif hour == 11 or hour == 13:
                    score += 10
                else:
                    score += 5
        
        # 3. 开板次数 (20分) - 越少越好
        if stock.open_times == 0:
            score += 20  # 一字板
        elif stock.open_times == 1:
            score += 15
        elif stock.open_times == 2:
            score += 10
        elif stock.open_times <= 3:
            score += 5
        
        # 4. 换手率 (15分) - 5-15%最佳
        if 5 <= stock.turnover_rate <= 15:
            score += 15
        elif 3 <= stock.turnover_rate < 5 or 15 < stock.turnover_rate <= 20:
            score += 10
        elif stock.turnover_rate < 3:
            score += 5  # 换手不足
        
        # 5. 板块热度 (10分)
        if stock.sector_limit_count >= 10:
            score += 10
        elif stock.sector_limit_count >= 5:
            score += 8
        elif stock.sector_limit_count >= 3:
            score += 5
        elif stock.sector_limit_count >= 1:
            score += 3
        
        # 6. 连板天数 (10分) - 连板有溢价
        if stock.consecutive_days >= 5:
            score += 10
        elif stock.consecutive_days >= 3:
            score += 8
        elif stock.consecutive_days >= 2:
            score += 6
        elif stock.consecutive_days == 1:
            score += 3
        
        return min(score, 100)
    
    def calculate_confidence(self, stock: LimitUpStock) -> float:
        """
        计算置信度(0-1)(借鉴advanced-ak-pack)
        
        综合:
        - 收盘价接近最高价 (40%)
        - 早盘VWAP斜率 (30%)
        - 板块热度 (30%)
        """
        # 1. 收盘接近度 (收盘越接近最高越强)
        close_high_prox = 1 - ((stock.high - stock.close) / stock.high) if stock.high > 0 else 0
        
        # 2. VWAP斜率排名 (归一化到0-1)
        vwap_score = 0.5  # 默认中性
        if stock.vwap_slope_morning is not None and np.isfinite(stock.vwap_slope_morning):
            # 正斜率越大越好
            vwap_score = min(max(stock.vwap_slope_morning * 100, 0), 1)
        
        # 3. 板块热度 (归一化)
        sector_heat = min(stock.sector_limit_count / 10, 1) if stock.sector_limit_count > 0 else 0
        
        confidence = (
            0.4 * close_high_prox +
            0.3 * vwap_score +
            0.3 * sector_heat
        )
        
        return max(0, min(confidence, 1))
    
    def select_qualified_stocks(
        self,
        candidates: List[LimitUpStock],
        min_quality_score: float = 70.0,
        min_confidence: float = 0.5,
        max_open_times: int = 2,
        min_seal_ratio: float = 0.03,
        prefer_first_board: bool = True,
        prefer_sector_leader: bool = True,
        check_market_timing: bool = True,
        current_date: Optional[str] = None
    ) -> List[LimitUpStock]:
        """
        筛选优质涨停股
        
        Args:
            candidates: 候选涨停股列表
            min_quality_score: 最低质量分
            min_confidence: 最低置信度
            max_open_times: 最多开板次数
            min_seal_ratio: 最低封单比例
            prefer_first_board: 优先首板
            prefer_sector_leader: 优先龙头
            check_market_timing: 是否检查市场择时
            current_date: 当前日期，用于市场择时检查
            
        Returns:
            筛选后的股票列表(按质量分排序)
        """
        # 第一步：市场择时检查（顶层风控）
        if check_market_timing:
            if current_date is None:
                current_date = datetime.now().strftime('%Y-%m-%d')
            
            if not is_safe_environment(current_date):
                self.logger.warning(f"市场环境不安全 ({current_date})，不建议操作")
                # 市场不安全时，返回空列表或只返回极少数最优质的股票
                # 这里选择返回空列表，完全避险
                return []
        
        qualified = []
        
        for stock in candidates:
            # 第二步：强化的烂板过滤（一票否决）
            # 开板次数超过2次的直接剔除，不管其他指标多好
            if stock.open_times > 2:
                self.logger.info(f"剔除烂板股票 {stock.symbol}: 开板次数={stock.open_times}")
                continue
            
            # 特殊情况：开板2次但封单弱的也剔除
            if stock.open_times == 2 and stock.seal_ratio < 0.05:
                self.logger.info(f"剔除弱势板 {stock.symbol}: 开板2次且封单比={stock.seal_ratio:.3f}")
                continue
            
            # 涨停时间过晚（14:00以后）且不是一字板的剔除
            if stock.limit_up_time and not stock.is_one_word:
                time_parts = stock.limit_up_time.split(':')
                if len(time_parts) >= 2:
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    if hour >= 14 and minute > 0:
                        self.logger.info(f"剔除尾盘板 {stock.symbol}: 涨停时间={stock.limit_up_time}")
                        continue
            
            # 第三步：基础条件筛选
            if (stock.quality_score >= min_quality_score and
                stock.confidence >= min_confidence and
                stock.seal_ratio >= min_seal_ratio):
                
                # 加分项
                bonus = 0
                if prefer_first_board and stock.is_first_board:
                    bonus += 5
                if prefer_sector_leader and stock.is_sector_leader:
                    bonus += 5
                
                # 对于零开板的强势股额外加分
                if stock.open_times == 0:
                    bonus += 10
                
                # 调整后的质量分
                adjusted_score = stock.quality_score + bonus
                
                qualified.append((adjusted_score, stock))
        
        # 按调整后质量分排序
        qualified.sort(key=lambda x: x[0], reverse=True)
        
        result = [stock for _, stock in qualified]
        
        self.logger.info(f"筛选完成: {len(candidates)}只候选 -> {len(result)}只合格")
        
        return result


if __name__ == "__main__":
    # 测试用例
    selector = EnhancedLimitUpSelector()
    
    # 模拟数据
    test_df = pd.DataFrame({
        "date": pd.date_range("2025-01-13", periods=5),
        "open": [10.0, 11.0, 12.1, 13.3, 14.6],
        "high": [11.0, 12.1, 13.3, 14.6, 16.0],
        "low": [9.8, 10.9, 12.0, 13.2, 14.5],
        "close": [11.0, 12.1, 13.3, 14.6, 14.5],
        "volume": [1000000] * 5
    })
    
    # 测试首板识别
    result = selector.detect_first_board(test_df, "000001")
    print("\n首板识别结果:")
    print(result[["date", "close", "is_limit_up", "prev_limit_up", "is_first_board"]])
    
    # 测试连板计算
    consecutive = selector.calculate_consecutive_days(result, "000001")
    print(f"\n连板天数: {consecutive}")
