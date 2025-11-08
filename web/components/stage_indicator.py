"""
智能交易阶段识别组件
自动识别当前所处的T日/T+1/T+2阶段，提供上下文建议
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

# 导入Phase 3颜色编码系统
try:
    from .color_scheme import Colors, Emojis, get_stage_color, get_stage_emoji
except ImportError:
    # 如果导入失败，使用默认值
    Colors = None
    Emojis = None
    get_stage_color = None
    get_stage_emoji = None


class StageIndicator:
    """交易阶段智能识别器"""
    
    # 阶段定义
    STAGE_T_DAY = "T日选股"
    STAGE_T1_AUCTION = "T+1竞价监控"
    STAGE_T1_TRADING = "T+1盘中交易"
    STAGE_T2_SELL = "T+2卖出决策"
    
    # 时间段定义
    MARKET_OPEN = (9, 30)
    MARKET_CLOSE = (15, 0)
    AUCTION_START = (9, 15)
    AUCTION_END = (9, 25)
    
    def __init__(self):
        """初始化阶段识别器"""
        self.now = datetime.now()
        self.hour = self.now.hour
        self.minute = self.now.minute
        
    def get_current_stage(self) -> Tuple[str, str, str]:
        """
        获取当前交易阶段
        
        Returns:
            (stage_name, description, suggestion): 阶段名称、描述、建议
        """
        # 9:15之前 - T日选股阶段
        if self.hour < 9 or (self.hour == 9 and self.minute < 15):
            return (
                self.STAGE_T_DAY,
                "盘前准备阶段",
                "筛选今日涨停股，构建T+1监控池"
            )
        
        # 9:15-9:25 - 集合竞价监控
        elif self.hour == 9 and 15 <= self.minute < 25:
            return (
                self.STAGE_T1_AUCTION,
                "集合竞价监控中",
                "重点关注候选池竞价表现，准备买入决策"
            )
        
        # 9:25-9:30 - 竞价结果分析
        elif self.hour == 9 and 25 <= self.minute < 30:
            return (
                self.STAGE_T1_AUCTION,
                "竞价结果分析",
                "快速评估竞价结果，确定最终买入名单"
            )
        
        # 9:30-15:00 - 盘中交易
        elif (self.hour == 9 and self.minute >= 30) or (9 < self.hour < 15):
            return (
                self.STAGE_T1_TRADING,
                "盘中交易时段",
                "执行买入决策，关注持仓变化"
            )
        
        # 15:00之后 - T+2准备/T日选股
        else:
            return (
                self.STAGE_T2_SELL,
                "盘后复盘阶段",
                "复盘今日交易，准备明日卖出策略"
            )
    
    def get_countdown(self) -> Dict[str, Any]:
        """
        获取倒计时信息
        
        Returns:
            包含倒计时信息的字典
        """
        stage, _, _ = self.get_current_stage()
        
        if stage == self.STAGE_T1_AUCTION and self.minute < 25:
            # 竞价期间，倒计时到9:25
            target = self.now.replace(hour=9, minute=25, second=0, microsecond=0)
            delta = target - self.now
            return {
                "show": True,
                "target": "开盘",
                "seconds": int(delta.total_seconds()),
                "display": f"{delta.seconds // 60}分{delta.seconds % 60}秒"
            }
        
        elif stage == self.STAGE_T1_AUCTION and self.minute >= 25:
            # 9:25-9:30，倒计时到开盘
            target = self.now.replace(hour=9, minute=30, second=0, microsecond=0)
            delta = target - self.now
            return {
                "show": True,
                "target": "开盘",
                "seconds": int(delta.total_seconds()),
                "display": f"{delta.seconds // 60}分{delta.seconds % 60}秒"
            }
        
        elif stage == self.STAGE_T1_TRADING and self.hour < 15:
            # 盘中，倒计时到收盘
            target = self.now.replace(hour=15, minute=0, second=0, microsecond=0)
            delta = target - self.now
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            return {
                "show": True,
                "target": "收盘",
                "seconds": int(delta.total_seconds()),
                "display": f"{hours}小时{minutes}分钟"
            }
        
        else:
            return {
                "show": False,
                "target": "",
                "seconds": 0,
                "display": ""
            }
    
    def get_stage_color(self) -> str:
        """获取当前阶段的颜色标识"""
        stage, _, _ = self.get_current_stage()
        
        # 使用Phase 3统一颜色系统
        if Emojis:
            color_map = {
                self.STAGE_T_DAY: Emojis.BLUE_CIRCLE,  # 蓝色 - 准备
                self.STAGE_T1_AUCTION: Emojis.GREEN_CIRCLE,  # 绿色 - 关键时刻
                self.STAGE_T1_TRADING: Emojis.YELLOW_CIRCLE,  # 黄色 - 执行中
                self.STAGE_T2_SELL: f"{Emojis.SELL}",  # 卖出图标 - 收尾
            }
        else:
            # 回退到原始颜色
            color_map = {
                self.STAGE_T_DAY: "🔵",
                self.STAGE_T1_AUCTION: "🟢",
                self.STAGE_T1_TRADING: "🟡",
                self.STAGE_T2_SELL: "🟣",
            }
        
        return color_map.get(stage, "⚪")
    
    def render(self):
        """渲染阶段指示器"""
        stage, description, suggestion = self.get_current_stage()
        countdown = self.get_countdown()
        color = self.get_stage_color()
        
        # 创建一个醒目的提示框（使用Phase 3样式）
        bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        if Colors:
            # 根据阶段使用不同的渐变色
            if stage == self.STAGE_T1_AUCTION:
                bg_gradient = f"linear-gradient(135deg, {Colors.SUCCESS} 0%, {Colors.MEDIUM_GREEN} 100%)"
            elif stage == self.STAGE_T1_TRADING:
                bg_gradient = f"linear-gradient(135deg, {Colors.WARNING} 0%, {Colors.LIGHT_ORANGE} 100%)"
        
        st.markdown(f"""
        <div style="
            background: {bg_gradient};
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h2 style="margin: 0; font-size: 24px;">
                        {color} {stage}
                    </h2>
                    <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">
                        {description}
                    </p>
                </div>
                {f'''
                <div style="text-align: right;">
                    <div style="font-size: 12px; opacity: 0.8;">距离{countdown['target']}</div>
                    <div style="font-size: 28px; font-weight: bold;">{countdown['display']}</div>
                </div>
                ''' if countdown['show'] else ''}
            </div>
            <div style="
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid rgba(255,255,255,0.3);
            ">
                <div style="font-size: 12px; opacity: 0.8;">💡 当前建议</div>
                <div style="font-size: 16px; margin-top: 5px;">
                    {suggestion}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def get_stage_tips(self, data: Dict[str, Any] = None) -> list:
        """
        根据当前阶段和数据提供智能提示
        
        Args:
            data: 相关数据，用于生成个性化提示
            
        Returns:
            提示列表
        """
        stage, _, _ = self.get_current_stage()
        data = data or {}
        
        tips_map = {
            self.STAGE_T_DAY: self._get_t_day_tips(data),
            self.STAGE_T1_AUCTION: self._get_t1_auction_tips(data),
            self.STAGE_T1_TRADING: self._get_t1_trading_tips(data),
            self.STAGE_T2_SELL: self._get_t2_sell_tips(data),
        }
        
        return tips_map.get(stage, [])
    
    def _get_t_day_tips(self, data: Dict) -> list:
        """T日选股阶段的提示"""
        tips = []
        
        limitup_count = data.get('limitup_count', 0)
        if limitup_count > 100:
            tips.append("💡 今日涨停数较多，市场情绪活跃，可适当放宽筛选条件")
        elif limitup_count < 30:
            tips.append("⚠️  今日涨停数较少，市场情绪低迷，建议提高筛选标准")
        
        candidate_count = data.get('candidate_count', 0)
        if candidate_count > 15:
            tips.append("⚠️  候选池数量较多，建议进一步筛选，聚焦核心标的")
        elif candidate_count == 0:
            tips.append("⚠️  当前无候选股票，请调整筛选条件")
        
        return tips or ["💡 开始筛选今日涨停股，构建明日监控池"]
    
    def _get_t1_auction_tips(self, data: Dict) -> list:
        """T+1竞价监控阶段的提示"""
        tips = []
        
        strong_count = data.get('strong_count', 0)
        weak_count = data.get('weak_count', 0)
        
        if strong_count > 0:
            tips.append(f"💡 {strong_count}只候选股竞价强势，建议优先买入")
        
        if weak_count > 0:
            tips.append(f"⚠️  {weak_count}只候选股竞价走弱，建议放弃")
        
        return tips or ["💡 重点关注竞价涨幅 >5% 的候选股"]
    
    def _get_t1_trading_tips(self, data: Dict) -> list:
        """T+1盘中交易阶段的提示"""
        tips = []
        
        position_count = data.get('position_count', 0)
        if position_count > 0:
            tips.append(f"✅ 当前持仓 {position_count} 只，关注盘中走势")
        else:
            tips.append("💡 当前无持仓，可关注盘中低吸机会")
        
        return tips
    
    def _get_t2_sell_tips(self, data: Dict) -> list:
        """T+2卖出阶段的提示"""
        tips = []
        
        profit_count = data.get('profit_count', 0)
        loss_count = data.get('loss_count', 0)
        
        if profit_count > 0:
            tips.append(f"✅ {profit_count}只持仓盈利，建议适时止盈")
        
        if loss_count > 0:
            tips.append(f"⚠️  {loss_count}只持仓亏损，注意止损")
        
        return tips or ["💡 复盘今日交易，准备明日策略"]


def render_stage_indicator(data: Dict[str, Any] = None):
    """
    渲染阶段指示器（便捷函数）
    
    Args:
        data: 相关数据，用于生成个性化提示
    """
    indicator = StageIndicator()
    indicator.render()
    
    # 显示智能提示
    tips = indicator.get_stage_tips(data)
    if tips:
        st.markdown("### 💡 智能提示")
        for tip in tips:
            st.info(tip)


# 用于测试
if __name__ == "__main__":
    st.set_page_config(page_title="阶段指示器测试", layout="wide")
    
    st.title("交易阶段智能识别测试")
    
    # 模拟数据
    test_data = {
        'limitup_count': 85,
        'candidate_count': 8,
        'strong_count': 3,
        'weak_count': 2,
        'position_count': 5,
        'profit_count': 3,
        'loss_count': 1
    }
    
    render_stage_indicator(test_data)
    
    # 显示当前时间
    st.markdown("---")
    st.write(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
