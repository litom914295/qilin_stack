
from dataclasses import dataclass
import pandas as pd, numpy as np

@dataclass
class AgentResult:
    score: float
    tags: dict

class BaseAgent:
    name='base'
    def run(self, symbol, ctx): ...

class ZTQualityAgent(BaseAgent):
    name='zt_quality'
    def run(self, s, ctx):
        d: pd.DataFrame = ctx['ohlc']
        if len(d)<2: return AgentResult(0.5,{})
        last = d.iloc[-1]; prev = d.iloc[-2]
        up_ratio = (last['close']-prev['close'])/max(1e-6, prev['close'])
        low_shadow = (last['open']-last['low'])/max(1e-6, last['close'])
        sc = 0.55 + 0.2*(up_ratio>0.07) + 0.05*(low_shadow>0.01)
        return AgentResult(float(np.clip(sc,0,1)), {'up_ratio':float(up_ratio)} )

class LeaderAgent(BaseAgent):
    """龙头股识别Agent"""
    name = 'leader'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """分析股票是否具有龙头特征，融合题材热度与龙虎榜净买入（如有）。"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 20:
                return AgentResult(0.5, {'error': 'insufficient_data'})
            
            # 相对强度
            returns = d['close'].pct_change()
            relative_strength = returns.iloc[-20:].mean() / returns.std() if returns.std() > 0 else 0
            rs_term = min(1.0, max(0.0, relative_strength + 0.5))
            
            # 成交量比率
            volume_ma5 = d['volume'].rolling(5).mean()
            volume_ma20 = d['volume'].rolling(20).mean()
            volume_ratio = (volume_ma5.iloc[-1] / volume_ma20.iloc[-1]) if volume_ma20.iloc[-1] > 0 else 1
            vol_term = min(1.0, volume_ratio / 2)
            
            # 价格位置
            high_20d = d['high'].iloc[-20:].max()
            low_20d = d['low'].iloc[-20:].min()
            current_price = d['close'].iloc[-1]
            price_position = (current_price - low_20d) / (high_20d - low_20d) if high_20d > low_20d else 0.5
            
            # 题材热度（可选）
            concept_heat = float(ctx.get('concept_heat', 0.0) or 0.0)
            concept_term = float(np.clip(concept_heat / 10.0, 0.0, 1.0))
            
            # 龙虎榜净买入（可选）
            lhb_netbuy = float(ctx.get('lhb_netbuy', 0.0) or 0.0)
            lhb_term = float(np.clip(0.5 + np.tanh(lhb_netbuy / 1e7) * 0.25, 0.0, 1.0))
            
            # 综合评分（新增题材与龙虎榜两个维度）
            score = (
                0.35 * rs_term +
                0.25 * vol_term +
                0.20 * price_position +
                0.10 * concept_term +
                0.10 * lhb_term
            )
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'is_leader': score > 0.7,
                    'relative_strength': float(relative_strength),
                    'volume_ratio': float(volume_ratio),
                    'price_position': float(price_position),
                    'concept_heat': float(concept_heat),
                    'lhb_netbuy': float(lhb_netbuy)
                }
        except Exception as e:
            return AgentResult(0.5, {'error': str(e)})

class LHBSeatAgent(BaseAgent):
    name='lhb'
    def run(self, s, ctx):
        nb = float(ctx.get('lhb_netbuy',0.0))
        sc = 0.5 + np.tanh(nb/1e7)*0.2
        return AgentResult(float(np.clip(sc,0,1)), {'netbuy':nb})

class NewsAgent(BaseAgent):
    name='news'
    def run(self, s, ctx):
        n = len(ctx.get('news_titles',[]))
        sc = 0.5 + min(0.1, 0.02*n)
        return AgentResult(float(np.clip(sc,0,1)), {'n_news':n})

class ChipAgent(BaseAgent):
    """筹码分析Agent"""
    name = 'chip'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """分析筹码分布情况"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 30:
                return AgentResult(0.5, {'error': 'insufficient_data'})
            
            # 计算成本分布
            prices = d['close'].iloc[-30:]
            volumes = d['volume'].iloc[-30:]
            
            # 加权平均成本
            vwap = (prices * volumes).sum() / volumes.sum() if volumes.sum() > 0 else prices.mean()
            current_price = prices.iloc[-1]
            
            # 筹码集中度（用标准差衡量）
            price_std = prices.std()
            concentration = 1 / (1 + price_std / prices.mean()) if prices.mean() > 0 else 0.5
            
            # 获利盘比例
            profit_ratio = len(prices[prices < current_price]) / len(prices)
            
            # 筹码稳定性（低换手率表示筹码稳定）
            avg_volume = volumes.mean()
            recent_volume = volumes.iloc[-5:].mean()
            stability = 1 - min(1, recent_volume / avg_volume) if avg_volume > 0 else 0.5
            
            # 综合评分
            score = (
                0.3 * concentration +  # 集中度权重30%
                0.4 * profit_ratio +   # 获利盘权重40%
                0.3 * stability        # 稳定性权重30%
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'vwap': float(vwap),
                    'concentration': float(concentration),
                    'profit_ratio': float(profit_ratio),
                    'stability': float(stability)
                }
        except Exception as e:
            return AgentResult(0.5, {'error': str(e)})

class ChanAgent(BaseAgent):
    """缠论技术分析Agent"""
    name = 'chan'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """基于缠论进行技术分析"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 10:
                return AgentResult(0.5, {'error': 'insufficient_data'})
            
            # 简化的缠论分析：识别顶底分型
            highs = d['high'].values
            lows = d['low'].values
            
            # 寻找顶分型（中间K线高点最高）
            top_patterns = 0
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    top_patterns += 1
            
            # 寻找底分型（中间K线低点最低）
            bottom_patterns = 0
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    bottom_patterns += 1
            
            # 计算当前位置（靠近顶部还是底部）
            recent_high = highs[-5:].max()
            recent_low = lows[-5:].min()
            current_price = d['close'].iloc[-1]
            position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            # 趋势判断（简单均线）
            ma5 = d['close'].rolling(5).mean().iloc[-1]
            ma20 = d['close'].rolling(20).mean().iloc[-1] if len(d) >= 20 else ma5
            trend_score = 0.5 + 0.3 * np.sign(ma5 - ma20)
            
            # 综合评分
            pattern_score = 0.5 + 0.1 * (bottom_patterns - top_patterns)
            score = 0.5 * pattern_score + 0.3 * (1 - position) + 0.2 * trend_score
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'top_patterns': int(top_patterns),
                    'bottom_patterns': int(bottom_patterns),
                    'position': float(position),
                    'trend': 'up' if ma5 > ma20 else 'down'
                }
        except Exception as e:
            return AgentResult(0.5, {'error': str(e)})

class ElliottAgent(BaseAgent):
    """艾略特波浪分析Agent"""
    name = 'elliott'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """基于艾略特波浪理论进行分析"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 50:
                return AgentResult(0.5, {'error': 'insufficient_data'})
            
            prices = d['close'].values
            
            # 简化的波浪识别：寻找5浪上升或3浪下跌
            # 这里使用简单的高低点来模拟波浪
            peaks = []
            troughs = []
            
            for i in range(5, len(prices) - 5):
                if prices[i] == max(prices[i-5:i+6]):
                    peaks.append(i)
                if prices[i] == min(prices[i-5:i+6]):
                    troughs.append(i)
            
            # 判断当前可能处于哪个波浪
            wave_score = 0.5
            if len(peaks) >= 2 and len(troughs) >= 2:
                # 上升趋势：低点逐渐抬高
                if len(troughs) >= 2 and prices[troughs[-1]] > prices[troughs[-2]]:
                    wave_score = 0.7
                # 下降趋势：高点逐渐降低
                elif len(peaks) >= 2 and prices[peaks[-1]] < prices[peaks[-2]]:
                    wave_score = 0.3
            
            # 斐波那契回撤水平
            high_price = prices[-20:].max()
            low_price = prices[-20:].min()
            current_price = prices[-1]
            
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            fib_distances = [abs(current_price - (low_price + level * (high_price - low_price))) 
                           for level in fib_levels]
            min_distance = min(fib_distances)
            fib_score = 1 - (min_distance / (high_price - low_price)) if high_price > low_price else 0.5
            
            # 综合评分
            score = 0.6 * wave_score + 0.4 * fib_score
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'wave_count': len(peaks) + len(troughs),
                    'trend': 'up' if wave_score > 0.5 else 'down',
                    'fib_support': float(fib_score)
                }
        except Exception as e:
            return AgentResult(0.5, {'error': str(e)})

class FibAgent(BaseAgent):
    """斐波那契分析Agent"""
    name = 'fib'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """基于斐波那契水平进行支撑阻力分析"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 30:
                return AgentResult(0.5, {'error': 'insufficient_data'})
            
            # 获取近期高低点
            period = min(30, len(d))
            high = d['high'].iloc[-period:].max()
            low = d['low'].iloc[-period:].min()
            current = d['close'].iloc[-1]
            
            if high <= low:
                return AgentResult(0.5, {'error': 'invalid_range'})
            
            # 计算斐波那契水平
            fib_levels = {
                0.0: low,
                0.236: low + 0.236 * (high - low),
                0.382: low + 0.382 * (high - low),
                0.5: low + 0.5 * (high - low),
                0.618: low + 0.618 * (high - low),
                0.786: low + 0.786 * (high - low),
                1.0: high
            }
            
            # 找出最近的斐波那契水平
            closest_level = None
            min_distance = float('inf')
            for level, price in fib_levels.items():
                distance = abs(current - price)
                if distance < min_distance:
                    min_distance = distance
                    closest_level = level
            
            # 判断支撑或阻力
            support_score = 0.5
            if current > fib_levels[0.618]:  # 强势区域
                support_score = 0.7
            elif current > fib_levels[0.5]:  # 中性偏强
                support_score = 0.6
            elif current > fib_levels[0.382]:  # 中性偏弱
                support_score = 0.4
            else:  # 弱势区域
                support_score = 0.3
            
            # 计算与关键水平的距离分数
            distance_score = 1 - (min_distance / (high - low))
            
            # 综合评分
            score = 0.7 * support_score + 0.3 * distance_score
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'closest_fib_level': float(closest_level),
                    'support_strength': float(support_score),
                    'price_position': float((current - low) / (high - low))
                }
        except Exception as e:
            return AgentResult(0.5, {'error': str(e)})

class MarketMoodAgent(BaseAgent):
    """市场情绪分析Agent"""
    name = 'mood'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """分析市场整体情绪"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 5:
                return AgentResult(0.5, {'error': 'insufficient_data'})
            
            # 成交量情绪（成交量放大表示情绪高涨）
            volume_ma = d['volume'].rolling(10).mean() if len(d) >= 10 else d['volume'].mean()
            recent_volume = d['volume'].iloc[-5:].mean()
            volume_sentiment = min(1.0, recent_volume / volume_ma) if volume_ma > 0 else 0.5
            
            # 价格波动情绪（波动率）
            returns = d['close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0
            volatility_sentiment = 1 / (1 + volatility * 10)  # 高波动降低情绪分数
            
            # 涨跌比例
            up_days = len(returns[returns > 0])
            down_days = len(returns[returns < 0])
            total_days = up_days + down_days
            up_ratio = up_days / total_days if total_days > 0 else 0.5
            
            # 动量情绪
            momentum = returns.iloc[-5:].mean() if len(returns) >= 5 else 0
            momentum_sentiment = 0.5 + np.tanh(momentum * 100) * 0.5
            
            # 综合市场情绪
            score = (
                0.25 * volume_sentiment +
                0.25 * volatility_sentiment +
                0.25 * up_ratio +
                0.25 * momentum_sentiment
            
            # 判断市场情绪类型
            if score > 0.7:
                mood_type = 'euphoric'
            elif score > 0.55:
                mood_type = 'optimistic'
            elif score > 0.45:
                mood_type = 'neutral'
            elif score > 0.3:
                mood_type = 'pessimistic'
            else:
                mood_type = 'fearful'
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'mood_type': mood_type,
                    'volume_sentiment': float(volume_sentiment),
                    'volatility': float(volatility),
                    'up_ratio': float(up_ratio),
                    'momentum': float(momentum)
                }
        except Exception as e:
            return AgentResult(0.5, {'error': str(e)})

class RiskGuardAgent(BaseAgent):
    """风险控制Agent"""
    name = 'risk'
    
    def run(self, symbol: str, ctx: dict) -> AgentResult:
        """评估交易风险"""
        try:
            d = ctx.get('ohlc', pd.DataFrame())
            if d.empty or len(d) < 20:
                return AgentResult(0.3, {'error': 'insufficient_data', 'risk': 'high'})
            
            # 计算各种风险指标
            returns = d['close'].pct_change().dropna()
            
            # 1. 波动率风险
            volatility = returns.std()
            volatility_risk = min(1.0, volatility * 20)  # 标准化到0-1
            
            # 2. 下跌风险（最大回撤）
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            drawdown_risk = min(1.0, max_drawdown * 5)
            
            # 3. 流动性风险
            avg_volume = d['volume'].mean()
            recent_volume = d['volume'].iloc[-5:].mean()
            liquidity_risk = 1 - min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
            
            # 4. 趋势反转风险
            if len(d) >= 20:
                ma5 = d['close'].rolling(5).mean().iloc[-1]
                ma20 = d['close'].rolling(20).mean().iloc[-1]
                trend_risk = 0.3 if ma5 > ma20 else 0.7  # 上升趋势风险较低
            else:
                trend_risk = 0.5
            
            # 5. 价格极端风险（布林带）
            if len(d) >= 20:
                sma = d['close'].rolling(20).mean().iloc[-1]
                std = d['close'].rolling(20).std().iloc[-1]
                upper_band = sma + 2 * std
                lower_band = sma - 2 * std
                current_price = d['close'].iloc[-1]
                
                if current_price > upper_band:
                    extreme_risk = 0.8  # 超买风险
                elif current_price < lower_band:
                    extreme_risk = 0.7  # 超卖但仍有风险
                else:
                    extreme_risk = 0.3
            else:
                extreme_risk = 0.5
            
            # 综合风险评分（风险越高，分数越低）
            total_risk = (
                0.2 * volatility_risk +
                0.25 * drawdown_risk +
                0.15 * liquidity_risk +
                0.2 * trend_risk +
                0.2 * extreme_risk
            
            # 风险守护分数（反向）
            score = 1 - total_risk
            
            # 风险等级
            if total_risk < 0.3:
                risk_level = 'low'
            elif total_risk < 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return AgentResult(
                score=float(np.clip(score, 0, 1)),
                tags={
                    'risk_level': risk_level,
                    'volatility_risk': float(volatility_risk),
                    'drawdown_risk': float(drawdown_risk),
                    'liquidity_risk': float(liquidity_risk),
                    'trend_risk': float(trend_risk),
                    'extreme_risk': float(extreme_risk),
                    'total_risk': float(total_risk)
                }
        except Exception as e:
            return AgentResult(0.3, {'error': str(e), 'risk': 'unknown'})
