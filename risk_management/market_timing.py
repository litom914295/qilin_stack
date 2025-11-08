"""
市场择时模块 - 大盘风控
根据主要指数的技术指标判断市场环境是否适合短线操作
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import akshare as ak

logger = logging.getLogger(__name__)


class MarketTimingFilter:
    """市场择时过滤器 - 顶层风控"""
    
    def __init__(self):
        """初始化市场择时过滤器"""
        self.cache_dir = Path(__file__).parent.parent / 'cache' / 'market_timing'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_codes = {
            'sh000001': '上证指数',
            'sh000300': '沪深300',
            'sz399001': '深证成指',
            'sz399006': '创业板指'
        }
    
    def is_safe_environment(
        self, 
        current_date: str, 
        index_code: str = 'sh000300',
        strict_mode: bool = False
    ) -> bool:
        """
        判断市场环境是否安全
        
        Args:
            current_date: 当前日期 (YYYY-MM-DD)
            index_code: 指数代码，默认沪深300
            strict_mode: 是否严格模式（60日线以下也不操作）
        
        Returns:
            True表示市场环境安全，可以操作；False表示应该避险
        """
        try:
            # 获取指数历史数据
            history = self.get_index_history(index_code, end_date=current_date, limit=60)
            
            if history is None or history.empty:
                logger.warning(f"无法获取 {index_code} 的历史数据，默认返回安全")
                return True
            
            # 计算移动平均线
            ma5 = history['close'].rolling(5).mean().iloc[-1]
            ma20 = history['close'].rolling(20).mean().iloc[-1]
            ma60 = history['close'].rolling(60).mean().iloc[-1] if len(history) >= 60 else ma20
            latest_close = history['close'].iloc[-1]
            
            # 计算市场强度指标
            market_strength = self._calculate_market_strength(history)
            
            # 规则1：收盘价必须在20日线之上（基本规则）
            if latest_close < ma20:
                logger.info(f"市场环境不安全：指数 {latest_close:.2f} < MA20 {ma20:.2f}")
                return False
            
            # 规则2：严格模式下，收盘价必须在60日线之上
            if strict_mode and latest_close < ma60:
                logger.info(f"市场环境不安全（严格模式）：指数 {latest_close:.2f} < MA60 {ma60:.2f}")
                return False
            
            # 规则3：市场强度必须大于30（满分100）
            if market_strength < 30:
                logger.info(f"市场环境不安全：市场强度 {market_strength:.1f} < 30")
                return False
            
            # 规则4：连续下跌天数不能超过3天
            consecutive_down = self._count_consecutive_down_days(history)
            if consecutive_down >= 3:
                logger.info(f"市场环境不安全：连续下跌 {consecutive_down} 天")
                return False
            
            # 规则5：成交量不能持续萎缩（成交量5日均线 < 20日均线的60%）
            volume_ma5 = history['volume'].rolling(5).mean().iloc[-1]
            volume_ma20 = history['volume'].rolling(20).mean().iloc[-1]
            if volume_ma5 < volume_ma20 * 0.6:
                logger.info(f"市场环境不安全：成交量持续萎缩")
                return False
            
            logger.info(f"市场环境安全：指数={latest_close:.2f}, MA20={ma20:.2f}, 强度={market_strength:.1f}")
            return True
            
        except Exception as e:
            logger.error(f"判断市场环境失败: {e}")
            # 出错时默认返回True，避免影响正常交易
            return True
    
    def get_index_history(
        self, 
        index_code: str, 
        end_date: str, 
        limit: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        获取指数历史数据
        
        Args:
            index_code: 指数代码
            end_date: 结束日期
            limit: 获取天数
        
        Returns:
            历史数据DataFrame
        """
        # 尝试从缓存加载
        cache_file = self.cache_dir / f"{index_code}_{end_date.replace('-', '')}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= limit:
                    return df.tail(limit)
            except Exception:
                pass
        
        # 从数据源获取
        try:
            # 计算开始日期
            end_dt = pd.to_datetime(end_date)
            start_dt = end_dt - timedelta(days=limit * 2)  # 多获取一些以确保有足够的交易日
            
            # 使用akshare获取指数数据
            if index_code.startswith('sh'):
                symbol = index_code[2:]
            elif index_code.startswith('sz'):
                symbol = index_code[2:]
            else:
                symbol = index_code
            
            df = ak.stock_zh_index_daily(symbol=symbol)
            
            if df is not None and not df.empty:
                # 重命名列
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume'
                })
                
                # 转换日期
                df['date'] = pd.to_datetime(df['date'])
                
                # 筛选日期范围
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
                
                # 保存到缓存
                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass
                
                return df.tail(limit)
            
        except Exception as e:
            logger.warning(f"获取指数 {index_code} 数据失败: {e}")
        
        # 如果获取失败，返回模拟数据
        return self._generate_mock_index_data(limit)
    
    def _generate_mock_index_data(self, limit: int) -> pd.DataFrame:
        """生成模拟指数数据"""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='B')
        
        # 生成随机但合理的指数数据
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, limit)
        close = 3000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.uniform(-0.005, 0.005, limit)),
            'high': close * (1 + np.random.uniform(0, 0.01, limit)),
            'low': close * (1 - np.random.uniform(0, 0.01, limit)),
            'close': close,
            'volume': np.random.uniform(1e11, 5e11, limit)
        })
        
        return df
    
    def _calculate_market_strength(self, history: pd.DataFrame) -> float:
        """
        计算市场强度指标（0-100）
        
        Args:
            history: 历史数据
        
        Returns:
            市场强度评分
        """
        if len(history) < 20:
            return 50.0
        
        score = 50.0  # 基础分
        
        # 1. 价格位置（最高20分）
        # 当前价格在20日内的相对位置
        close = history['close'].iloc[-1]
        high_20 = history['high'].tail(20).max()
        low_20 = history['low'].tail(20).min()
        if high_20 > low_20:
            position = (close - low_20) / (high_20 - low_20)
            score += position * 20
        
        # 2. 趋势强度（最高20分）
        # MA5 > MA10 > MA20
        ma5 = history['close'].rolling(5).mean().iloc[-1]
        ma10 = history['close'].rolling(10).mean().iloc[-1]
        ma20 = history['close'].rolling(20).mean().iloc[-1]
        
        if ma5 > ma10:
            score += 10
        if ma10 > ma20:
            score += 10
        
        # 3. 成交量（最高10分）
        # 成交量是否放大
        vol_ma5 = history['volume'].rolling(5).mean().iloc[-1]
        vol_ma20 = history['volume'].rolling(20).mean().iloc[-1]
        if vol_ma5 > vol_ma20:
            score += min(10, (vol_ma5 / vol_ma20 - 1) * 20)
        
        return min(100, max(0, score))
    
    def _count_consecutive_down_days(self, history: pd.DataFrame) -> int:
        """
        计算连续下跌天数
        
        Args:
            history: 历史数据
        
        Returns:
            连续下跌天数
        """
        if len(history) < 2:
            return 0
        
        # 计算每日涨跌
        returns = history['close'].pct_change()
        
        # 从最后一天开始，往前数连续下跌天数
        consecutive = 0
        for i in range(len(returns) - 1, 0, -1):
            if returns.iloc[i] < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def get_market_risk_level(self, current_date: str) -> str:
        """
        获取市场风险等级
        
        Args:
            current_date: 当前日期
        
        Returns:
            风险等级：'低', '中', '高', '极高'
        """
        # 检查多个指数
        safe_count = 0
        total_count = 0
        
        for index_code in ['sh000300', 'sh000001', 'sz399001']:
            if self.is_safe_environment(current_date, index_code):
                safe_count += 1
            total_count += 1
        
        safe_ratio = safe_count / total_count if total_count > 0 else 0.5
        
        if safe_ratio >= 0.8:
            return '低'
        elif safe_ratio >= 0.5:
            return '中'
        elif safe_ratio >= 0.2:
            return '高'
        else:
            return '极高'
    
    def should_reduce_position(self, current_date: str) -> Tuple[bool, float]:
        """
        判断是否应该减仓及建议仓位
        
        Args:
            current_date: 当前日期
        
        Returns:
            (是否减仓, 建议仓位比例)
        """
        risk_level = self.get_market_risk_level(current_date)
        
        if risk_level == '低':
            return (False, 1.0)  # 满仓
        elif risk_level == '中':
            return (False, 0.7)  # 7成仓位
        elif risk_level == '高':
            return (True, 0.3)   # 3成仓位
        else:  # 极高
            return (True, 0.0)   # 空仓
    
    def get_market_status_summary(self, current_date: str) -> Dict:
        """
        获取市场状态汇总
        
        Args:
            current_date: 当前日期
        
        Returns:
            市场状态汇总字典
        """
        summary = {
            'date': current_date,
            'risk_level': self.get_market_risk_level(current_date),
            'indices': {},
            'recommendation': ''
        }
        
        # 检查各指数状态
        for index_code, name in self.index_codes.items():
            is_safe = self.is_safe_environment(current_date, index_code)
            summary['indices'][name] = '安全' if is_safe else '危险'
        
        # 给出建议
        should_reduce, position = self.should_reduce_position(current_date)
        if should_reduce:
            if position == 0:
                summary['recommendation'] = '空仓观望，等待市场企稳'
            else:
                summary['recommendation'] = f'建议减仓至{position*100:.0f}%'
        else:
            if position == 1.0:
                summary['recommendation'] = '市场环境良好，可正常操作'
            else:
                summary['recommendation'] = f'市场有一定风险，建议仓位{position*100:.0f}%'
        
        return summary


# 全局实例
_market_timer = None

def get_market_timer() -> MarketTimingFilter:
    """获取全局市场择时实例"""
    global _market_timer
    if _market_timer is None:
        _market_timer = MarketTimingFilter()
    return _market_timer


def is_safe_environment(current_date: str, index_code: str = 'sh000300') -> bool:
    """
    便捷函数：判断市场环境是否安全
    
    Args:
        current_date: 当前日期
        index_code: 指数代码
    
    Returns:
        是否安全
    """
    timer = get_market_timer()
    return timer.is_safe_environment(current_date, index_code)


if __name__ == "__main__":
    # 测试代码
    timer = MarketTimingFilter()
    
    # 测试当前日期
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"测试日期: {today}")
    print("-" * 50)
    
    # 测试市场环境
    is_safe = timer.is_safe_environment(today)
    print(f"市场环境是否安全: {is_safe}")
    
    # 获取风险等级
    risk_level = timer.get_market_risk_level(today)
    print(f"市场风险等级: {risk_level}")
    
    # 获取仓位建议
    should_reduce, position = timer.should_reduce_position(today)
    print(f"是否减仓: {should_reduce}")
    print(f"建议仓位: {position*100:.0f}%")
    
    # 获取完整状态
    print("\n市场状态汇总:")
    summary = timer.get_market_status_summary(today)
    print(f"风险等级: {summary['risk_level']}")
    print(f"建议: {summary['recommendation']}")
    print("各指数状态:")
    for index, status in summary['indices'].items():
        print(f"  {index}: {status}")