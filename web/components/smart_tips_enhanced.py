"""
增强版智能提示系统
提供更丰富的场景化建议和风险预警
"""

import streamlit as st
import pandas as pd
from datetime import datetime, time
from typing import Dict, Any, List, Optional
from .color_scheme import Colors, Emojis, get_alert_box_html
from .smart_actions import SmartTipSystem


class EnhancedSmartTipSystem(SmartTipSystem):
    """增强版智能提示系统"""
    
    def __init__(self):
        super().__init__()
        self.risk_rules = self._init_risk_rules()
        self.market_sentiment_thresholds = {
            '极度亢奋': 150,
            '活跃': 100,
            '正常': 50,
            '低迷': 30,
            '冰点': 0
        }
    
    def _init_risk_rules(self) -> List[Dict]:
        """初始化风险规则"""
        return [
            {
                'name': '集中度风险',
                'condition': lambda data: data.get('sector_concentration', 0) > 60,
                'level': 'high',
                'message': '⚠️ 候选股集中在单一板块（占比>60%），存在板块轮动风险，建议分散'
            },
            {
                'name': '连板炸板风险',
                'condition': lambda data: data.get('failed_limitup_rate', 0) > 30,
                'level': 'high',
                'message': '🔴 今日炸板率>30%，市场分歧加剧，建议降低仓位或观望'
            },
            {
                'name': '新股上市风险',
                'condition': lambda data: data.get('new_stock_count', 0) > 5,
                'level': 'medium',
                'message': '💡 今日新股上市较多，可能分流资金，注意市场情绪变化'
            },
            {
                'name': '指数跳水风险',
                'condition': lambda data: data.get('index_change', 0) < -2,
                'level': 'high',
                'message': '📉 指数跌幅>2%，市场环境恶化，建议谨慎操作或空仓观望'
            },
            {
                'name': '成交量异常',
                'condition': lambda data: data.get('volume_ratio', 0) < 0.5,
                'level': 'medium',
                'message': '⚠️ 量能不足（量比<0.5），市场活跃度低，谨防假突破'
            },
            {
                'name': '情绪冰点',
                'condition': lambda data: data.get('limitup_count', 100) < 20,
                'level': 'high',
                'message': '❄️ 涨停数<20只，市场情绪冰点，建议空仓休息或等待转机'
            }
        ]
    
    def analyze_market_sentiment(self, limitup_count: int) -> Dict[str, Any]:
        """
        分析市场情绪
        
        Args:
            limitup_count: 涨停数量
            
        Returns:
            情绪分析结果
        """
        if limitup_count >= 150:
            sentiment = '极度亢奋'
            color = Colors.STRONG_GREEN
            emoji = f"{Emojis.FIRE}{Emojis.FIRE}{Emojis.FIRE}"
            advice = '市场情绪极度亢奋，注意追高风险，可适当降低仓位'
        elif limitup_count >= 100:
            sentiment = '活跃'
            color = Colors.SUCCESS
            emoji = f"{Emojis.GREEN_CIRCLE}{Emojis.STRONG}"
            advice = '市场情绪活跃，适合积极操作，可适当放宽筛选条件'
        elif limitup_count >= 50:
            sentiment = '正常'
            color = Colors.PRIMARY
            emoji = Emojis.NEUTRAL
            advice = '市场情绪正常，按照既定策略操作即可'
        elif limitup_count >= 30:
            sentiment = '低迷'
            color = Colors.WARNING
            emoji = Emojis.YELLOW_CIRCLE
            advice = '市场情绪低迷，建议提高筛选标准，减少操作频率'
        else:
            sentiment = '冰点'
            color = Colors.DANGER
            emoji = f"{Emojis.RED_CIRCLE}{Emojis.WARNING}"
            advice = '市场情绪冰点，建议空仓休息，等待市场转机'
        
        return {
            'sentiment': sentiment,
            'color': color,
            'emoji': emoji,
            'advice': advice,
            'score': min(100, int((limitup_count / 150) * 100))
        }
    
    def check_risk_warnings(self, data: Dict[str, Any]) -> List[Dict]:
        """
        检查风险预警
        
        Args:
            data: 数据字典
            
        Returns:
            风险警告列表
        """
        warnings = []
        
        for rule in self.risk_rules:
            try:
                if rule['condition'](data):
                    warnings.append({
                        'name': rule['name'],
                        'level': rule['level'],
                        'message': rule['message'],
                        'type': 'danger' if rule['level'] == 'high' else 'warning'
                    })
            except Exception as e:
                # 忽略规则检查错误
                pass
        
        return warnings
    
    def generate_sector_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成板块分析
        
        Args:
            df: 股票数据DataFrame（需包含'sector'列）
            
        Returns:
            板块分析结果
        """
        if df.empty or 'sector' not in df.columns:
            return {}
        
        sector_counts = df['sector'].value_counts()
        total = len(df)
        
        # 前三大板块
        top_sectors = []
        for i, (sector, count) in enumerate(sector_counts.head(3).items()):
            percentage = (count / total) * 100
            top_sectors.append({
                'sector': sector,
                'count': count,
                'percentage': percentage,
                'rank': i + 1
            })
        
        # 集中度分析
        max_percentage = (sector_counts.iloc[0] / total * 100) if len(sector_counts) > 0 else 0
        
        if max_percentage > 60:
            concentration_level = '高度集中'
            concentration_color = Colors.DANGER
            concentration_advice = '板块集中度过高，建议分散到其他板块'
        elif max_percentage > 40:
            concentration_level = '较为集中'
            concentration_color = Colors.WARNING
            concentration_advice = '板块分布较为集中，适度分散可降低风险'
        else:
            concentration_level = '均衡分散'
            concentration_color = Colors.SUCCESS
            concentration_advice = '板块分布均衡，风险分散合理'
        
        return {
            'top_sectors': top_sectors,
            'concentration_level': concentration_level,
            'concentration_color': concentration_color,
            'concentration_advice': concentration_advice,
            'max_percentage': max_percentage
        }
    
    def generate_timing_advice(self) -> Dict[str, str]:
        """
        生成时间相关的操作建议
        
        Returns:
            时间建议字典
        """
        now = datetime.now()
        current_time = now.time()
        
        # 定义关键时间点
        t_auction_start = time(9, 15)
        t_auction_end = time(9, 25)
        t_open = time(9, 30)
        t_morning_mid = time(10, 30)
        t_noon = time(11, 30)
        t_afternoon_start = time(13, 0)
        t_close = time(15, 0)
        
        if current_time < t_auction_start:
            return {
                'phase': '开盘前',
                'emoji': Emojis.CLOCK,
                'advice': '复盘昨日表现，准备今日监控池，关注隔夜消息面',
                'priority': '复盘分析'
            }
        elif t_auction_start <= current_time < t_auction_end:
            return {
                'phase': '竞价阶段',
                'emoji': Emojis.FIRE,
                'advice': '重点关注候选股竞价表现，涨幅>5%可考虑买入，跌幅>5%建议放弃',
                'priority': '竞价监控'
            }
        elif t_auction_end <= current_time < t_open:
            return {
                'phase': '集合竞价结束',
                'emoji': Emojis.TARGET,
                'advice': '最后确认买入标的，准备开盘挂单，注意流动性',
                'priority': '买入决策'
            }
        elif t_open <= current_time < t_morning_mid:
            return {
                'phase': '早盘',
                'emoji': Emojis.ROCKET,
                'advice': '观察个股开盘走势，强势股持有，弱势股止损',
                'priority': '盘中监控'
            }
        elif t_morning_mid <= current_time < t_noon:
            return {
                'phase': '午前',
                'emoji': Emojis.CHART,
                'advice': '评估上午走势，考虑是否调整持仓',
                'priority': '持仓调整'
            }
        elif t_noon <= current_time < t_afternoon_start:
            return {
                'phase': '午休',
                'emoji': '☕',
                'advice': '复盘上午走势，准备下午策略',
                'priority': '中场休息'
            }
        elif t_afternoon_start <= current_time < t_close:
            return {
                'phase': '下午盘',
                'emoji': Emojis.MONEY,
                'advice': 'T+2持仓考虑止盈/止损，关注尾盘资金流向',
                'priority': '卖出决策'
            }
        else:
            return {
                'phase': '收盘后',
                'emoji': '🌙',
                'advice': '统计今日收益，筛选明日候选池',
                'priority': '盘后选股'
            }
    
    def generate_performance_tips(self, performance_data: Dict) -> List[Dict]:
        """
        生成绩效相关提示
        
        Args:
            performance_data: 绩效数据
            
        Returns:
            提示列表
        """
        tips = []
        
        win_rate = performance_data.get('win_rate', 0)
        avg_profit = performance_data.get('avg_profit', 0)
        max_drawdown = performance_data.get('max_drawdown', 0)
        
        # 胜率分析
        if win_rate >= 70:
            tips.append({
                'type': 'success',
                'message': f"🏆 胜率 {win_rate:.1f}%，策略表现优秀，继续保持"
            })
        elif win_rate >= 50:
            tips.append({
                'type': 'info',
                'message': f"👍 胜率 {win_rate:.1f}%，策略表现正常"
            })
        else:
            tips.append({
                'type': 'warning',
                'message': f"⚠️ 胜率 {win_rate:.1f}%，需要反思策略或调整参数"
            })
        
        # 平均收益分析
        if avg_profit >= 5:
            tips.append({
                'type': 'success',
                'message': f"💰 平均收益 {avg_profit:+.2f}%，盈利能力强"
            })
        elif avg_profit >= 0:
            tips.append({
                'type': 'info',
                'message': f"💵 平均收益 {avg_profit:+.2f}%，维持盈利"
            })
        else:
            tips.append({
                'type': 'danger',
                'message': f"📉 平均收益 {avg_profit:+.2f}%，需要优化策略"
            })
        
        # 最大回撤分析
        if abs(max_drawdown) > 10:
            tips.append({
                'type': 'danger',
                'message': f"⚠️ 最大回撤 {max_drawdown:.2f}%，风险控制需加强"
            })
        elif abs(max_drawdown) > 5:
            tips.append({
                'type': 'warning',
                'message': f"💡 最大回撤 {max_drawdown:.2f}%，注意控制风险"
            })
        
        return tips
    
    def render_enhanced_tips(self, stage: str, data: Dict[str, Any]):
        """
        渲染增强版智能提示
        
        Args:
            stage: 当前交易阶段
            data: 数据字典
        """
        st.markdown("### 💡 智能提示与建议")
        
        # 1. 时间建议
        timing = self.generate_timing_advice()
        st.markdown(get_alert_box_html(
            f"{timing['emoji']} **{timing['phase']}** - {timing['advice']} | 当前重点: {timing['priority']}",
            'info'
        ), unsafe_allow_html=True)
        
        # 2. 市场情绪分析
        limitup_count = data.get('limitup_count', 0)
        if limitup_count > 0:
            sentiment = self.analyze_market_sentiment(limitup_count)
            st.markdown(f"""
            #### 📊 市场情绪: {sentiment['emoji']} {sentiment['sentiment']} ({sentiment['score']}分)
            """)
            st.progress(sentiment['score'] / 100)
            st.markdown(get_alert_box_html(
                sentiment['advice'],
                'success' if sentiment['score'] > 60 else ('warning' if sentiment['score'] > 30 else 'danger')
            ), unsafe_allow_html=True)
        
        # 3. 风险预警
        warnings = self.check_risk_warnings(data)
        if warnings:
            st.markdown("#### ⚠️ 风险预警")
            for warning in warnings:
                st.markdown(get_alert_box_html(
                    f"**{warning['name']}**: {warning['message']}",
                    warning['type']
                ), unsafe_allow_html=True)
        
        # 4. 基础提示（使用父类方法）
        basic_tips = self.generate_tips(stage, data)
        if basic_tips:
            st.markdown("#### 📝 操作建议")
            for tip in basic_tips:
                st.markdown(get_alert_box_html(tip['message'], tip['type']), unsafe_allow_html=True)
        
        # 5. 板块分析（如果有数据）
        if 'candidate_df' in data and not data['candidate_df'].empty:
            sector_analysis = self.generate_sector_analysis(data['candidate_df'])
            if sector_analysis:
                st.markdown("#### 🏢 板块分布")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    for sector_info in sector_analysis['top_sectors']:
                        st.markdown(f"""
                        - **{sector_info['sector']}**: {sector_info['count']}只 ({sector_info['percentage']:.1f}%)
                        """)
                
                with col2:
                    st.markdown(f"""
                    **集中度**: {sector_analysis['concentration_level']}  
                    {sector_analysis['concentration_advice']}
                    """)
        
        # 6. 绩效提示（如果有绩效数据）
        if 'performance_data' in data:
            perf_tips = self.generate_performance_tips(data['performance_data'])
            if perf_tips:
                st.markdown("#### 📈 策略绩效")
                for tip in perf_tips:
                    st.markdown(get_alert_box_html(tip['message'], tip['type']), unsafe_allow_html=True)


__all__ = ['EnhancedSmartTipSystem']
