"""
统一颜色编码系统
为整个涨停监控系统提供一致的颜色、图标和样式定义
"""

from typing import Literal, Tuple
from dataclasses import dataclass

# ==================== 颜色常量定义 ====================

class Colors:
    """颜色常量"""
    # 主题色
    PRIMARY = "#1f77b4"  # 主蓝色
    SECONDARY = "#ff7f0e"  # 次橙色
    
    # 状态色（核心颜色编码）
    SUCCESS = "#28a745"  # 🟢 绿色 - 强势/买入/持有
    WARNING = "#ffc107"  # 🟡 黄色 - 观望/等待/中性
    DANGER = "#dc3545"   # 🔴 红色 - 弱势/卖出/风险
    INACTIVE = "#6c757d" # ⚪ 灰色 - 未激活/已完成
    
    # 涨停强度色
    STRONG_GREEN = "#28a745"    # 极强
    MEDIUM_GREEN = "#5cb85c"    # 强势
    LIGHT_YELLOW = "#ffc107"    # 良好
    LIGHT_ORANGE = "#fd7e14"    # 观望
    MEDIUM_RED = "#dc3545"      # 走弱
    STRONG_RED = "#c82333"      # 弱势
    
    # 背景色
    BG_SUCCESS = "#d4edda"
    BG_WARNING = "#fff3cd"
    BG_DANGER = "#f8d7da"
    BG_INFO = "#d1ecf1"
    BG_LIGHT = "#f8f9fa"
    
    # 文字色
    TEXT_DARK = "#212529"
    TEXT_MUTED = "#6c757d"
    TEXT_LIGHT = "#f8f9fa"
    
    # 边框色
    BORDER_LIGHT = "#dee2e6"
    BORDER_DARK = "#495057"


class Emojis:
    """统一的Emoji图标"""
    # 状态指示
    STRONG = "💪"
    WEAK = "📉"
    NEUTRAL = "➖"
    WARNING = "⚠️"
    SUCCESS = "✅"
    ERROR = "❌"
    
    # 颜色圆点
    GREEN_CIRCLE = "🟢"
    YELLOW_CIRCLE = "🟡"
    RED_CIRCLE = "🔴"
    WHITE_CIRCLE = "⚪"
    BLUE_CIRCLE = "🔵"
    
    # 功能图标
    REFRESH = "🔄"
    SAVE = "💾"
    EXPORT = "📄"
    ALERT = "🔔"
    CHART = "📊"
    MONEY = "💰"
    SELL = "💸"
    BUY = "🛒"
    FILTER = "🔍"
    SETTINGS = "⚙️"
    
    # 交易阶段
    CLOCK = "🕐"
    FIRE = "🔥"
    TARGET = "🎯"
    ROCKET = "🚀"


@dataclass
class ThemeConfig:
    """主题配置"""
    font_size_title: str = "24px"
    font_size_subtitle: str = "18px"
    font_size_body: str = "14px"
    font_size_small: str = "12px"
    
    spacing_small: str = "8px"
    spacing_medium: str = "16px"
    spacing_large: str = "24px"
    
    border_radius: str = "8px"
    box_shadow: str = "0 2px 8px rgba(0,0,0,0.1)"


# ==================== 辅助函数 ====================

def get_strength_color(strength: float) -> str:
    """
    根据强度值返回对应颜色
    
    Args:
        strength: 强度值 0-10
        
    Returns:
        颜色代码
    """
    if strength >= 9:
        return Colors.STRONG_GREEN
    elif strength >= 7:
        return Colors.MEDIUM_GREEN
    elif strength >= 5:
        return Colors.LIGHT_YELLOW
    elif strength >= 3:
        return Colors.LIGHT_ORANGE
    elif strength >= 1:
        return Colors.MEDIUM_RED
    else:
        return Colors.STRONG_RED


def get_strength_emoji(strength: float) -> str:
    """
    根据强度值返回对应Emoji
    
    Args:
        strength: 强度值 0-10
        
    Returns:
        Emoji字符串
    """
    if strength >= 9:
        return f"{Emojis.GREEN_CIRCLE}{Emojis.STRONG}{Emojis.STRONG}{Emojis.STRONG}"
    elif strength >= 7:
        return f"{Emojis.GREEN_CIRCLE}{Emojis.STRONG}{Emojis.STRONG}"
    elif strength >= 5:
        return f"{Emojis.YELLOW_CIRCLE}{Emojis.STRONG}"
    elif strength >= 3:
        return f"{Emojis.YELLOW_CIRCLE}"
    elif strength >= 1:
        return f"{Emojis.RED_CIRCLE}"
    else:
        return f"{Emojis.RED_CIRCLE}{Emojis.WARNING}"


def get_strength_label(strength: float) -> str:
    """
    根据强度值返回文字描述
    
    Args:
        strength: 强度值 0-10
        
    Returns:
        强度描述
    """
    if strength >= 9:
        return "极强"
    elif strength >= 7:
        return "强势"
    elif strength >= 5:
        return "良好"
    elif strength >= 3:
        return "观望"
    elif strength >= 1:
        return "走弱"
    else:
        return "弱势"


def get_profit_color(profit_rate: float) -> str:
    """
    根据盈亏比例返回颜色
    
    Args:
        profit_rate: 盈亏比例（百分比）
        
    Returns:
        颜色代码
    """
    if profit_rate > 10:
        return Colors.STRONG_GREEN
    elif profit_rate > 0:
        return Colors.SUCCESS
    elif profit_rate > -5:
        return Colors.WARNING
    else:
        return Colors.DANGER


def get_profit_emoji(profit_rate: float) -> str:
    """
    根据盈亏比例返回Emoji
    
    Args:
        profit_rate: 盈亏比例（百分比）
        
    Returns:
        Emoji字符串
    """
    if profit_rate > 10:
        return f"{Emojis.GREEN_CIRCLE} {Emojis.ROCKET}"
    elif profit_rate > 0:
        return f"{Emojis.GREEN_CIRCLE}"
    elif profit_rate > -5:
        return f"{Emojis.YELLOW_CIRCLE}"
    else:
        return f"{Emojis.RED_CIRCLE} {Emojis.WARNING}"


def get_risk_level_config(profit_rate: float) -> Tuple[str, str, str]:
    """
    根据盈亏比例返回风险等级配置
    
    Args:
        profit_rate: 盈亏比例（百分比）
        
    Returns:
        (等级名称, 颜色, Emoji)
    """
    if profit_rate > 10:
        return "大幅盈利", Colors.STRONG_GREEN, Emojis.GREEN_CIRCLE
    elif profit_rate > 0:
        return "持有观察", Colors.SUCCESS, Emojis.GREEN_CIRCLE
    elif profit_rate > -5:
        return "谨慎观察", Colors.WARNING, Emojis.YELLOW_CIRCLE
    else:
        return "止损建议", Colors.DANGER, Emojis.RED_CIRCLE


def get_stage_color(stage: str) -> str:
    """
    根据交易阶段返回颜色
    
    Args:
        stage: 交易阶段 (T日/T+1/T+2)
        
    Returns:
        颜色代码
    """
    stage_colors = {
        "T日": Colors.PRIMARY,
        "T+1": Colors.SUCCESS,
        "T+2": Colors.WARNING,
        "盘后": Colors.INACTIVE,
    }
    return stage_colors.get(stage, Colors.INACTIVE)


def get_stage_emoji(stage: str) -> str:
    """
    根据交易阶段返回Emoji
    
    Args:
        stage: 交易阶段
        
    Returns:
        Emoji字符串
    """
    stage_emojis = {
        "T日选股": f"{Emojis.CHART}",
        "T+1竞价": f"{Emojis.FIRE}",
        "T+1交易": f"{Emojis.MONEY}",
        "T+2卖出": f"{Emojis.SELL}",
        "盘后": f"{Emojis.CLOCK}",
    }
    return stage_emojis.get(stage, Emojis.CLOCK)


# ==================== CSS样式生成器 ====================

def get_metric_card_style(
    bg_color: str = Colors.BG_LIGHT,
    border_color: str = Colors.BORDER_LIGHT,
    text_color: str = Colors.TEXT_DARK
) -> str:
    """
    生成指标卡片样式
    
    Args:
        bg_color: 背景色
        border_color: 边框色
        text_color: 文字色
        
    Returns:
        CSS样式字符串
    """
    theme = ThemeConfig()
    return f"""
        <style>
        .metric-card {{
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: {theme.border_radius};
            padding: {theme.spacing_medium};
            box-shadow: {theme.box_shadow};
            color: {text_color};
        }}
        .metric-title {{
            font-size: {theme.font_size_small};
            color: {Colors.TEXT_MUTED};
            margin-bottom: {theme.spacing_small};
        }}
        .metric-value {{
            font-size: {theme.font_size_title};
            font-weight: bold;
            margin: {theme.spacing_small} 0;
        }}
        .metric-delta {{
            font-size: {theme.font_size_body};
            margin-top: {theme.spacing_small};
        }}
        </style>
    """


def get_status_badge_html(
    label: str,
    status: Literal["success", "warning", "danger", "inactive"] = "success"
) -> str:
    """
    生成状态徽章HTML
    
    Args:
        label: 标签文字
        status: 状态类型
        
    Returns:
        HTML字符串
    """
    color_map = {
        "success": Colors.SUCCESS,
        "warning": Colors.WARNING,
        "danger": Colors.DANGER,
        "inactive": Colors.INACTIVE,
    }
    bg_map = {
        "success": Colors.BG_SUCCESS,
        "warning": Colors.BG_WARNING,
        "danger": Colors.BG_DANGER,
        "inactive": Colors.BG_LIGHT,
    }
    
    color = color_map.get(status, Colors.INACTIVE)
    bg = bg_map.get(status, Colors.BG_LIGHT)
    
    return f"""
        <span style="
            background-color: {bg};
            color: {color};
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            border: 1px solid {color};
        ">{label}</span>
    """


def get_progress_bar_html(
    value: float,
    max_value: float = 100,
    color: str = Colors.SUCCESS,
    height: str = "24px",
    show_label: bool = True
) -> str:
    """
    生成进度条HTML
    
    Args:
        value: 当前值
        max_value: 最大值
        color: 进度条颜色
        height: 进度条高度
        show_label: 是否显示标签
        
    Returns:
        HTML字符串
    """
    percentage = min(100, max(0, (value / max_value) * 100))
    label = f"{value:.1f}" if show_label else ""
    
    return f"""
        <div style="
            width: 100%;
            background-color: {Colors.BG_LIGHT};
            border-radius: 12px;
            overflow: hidden;
            height: {height};
            position: relative;
            border: 1px solid {Colors.BORDER_LIGHT};
        ">
            <div style="
                width: {percentage}%;
                background-color: {color};
                height: 100%;
                transition: width 0.3s ease;
            "></div>
            {f'<span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: bold; color: {Colors.TEXT_DARK};">{label}</span>' if show_label else ''}
        </div>
    """


def get_alert_box_html(
    message: str,
    alert_type: Literal["success", "warning", "danger", "info"] = "info"
) -> str:
    """
    生成警告框HTML
    
    Args:
        message: 消息内容
        alert_type: 警告类型
        
    Returns:
        HTML字符串
    """
    color_map = {
        "success": (Colors.SUCCESS, Colors.BG_SUCCESS),
        "warning": (Colors.WARNING, Colors.BG_WARNING),
        "danger": (Colors.DANGER, Colors.BG_DANGER),
        "info": (Colors.PRIMARY, Colors.BG_INFO),
    }
    emoji_map = {
        "success": Emojis.SUCCESS,
        "warning": Emojis.WARNING,
        "danger": Emojis.ERROR,
        "info": "ℹ️",
    }
    
    color, bg = color_map.get(alert_type, (Colors.PRIMARY, Colors.BG_INFO))
    emoji = emoji_map.get(alert_type, "ℹ️")
    
    return f"""
        <div style="
            background-color: {bg};
            border-left: 4px solid {color};
            padding: 12px 16px;
            border-radius: 4px;
            margin: 8px 0;
        ">
            <span style="font-size: 14px;">
                {emoji} {message}
            </span>
        </div>
    """


# ==================== 导出 ====================

__all__ = [
    'Colors',
    'Emojis',
    'ThemeConfig',
    'get_strength_color',
    'get_strength_emoji',
    'get_strength_label',
    'get_profit_color',
    'get_profit_emoji',
    'get_risk_level_config',
    'get_stage_color',
    'get_stage_emoji',
    'get_metric_card_style',
    'get_status_badge_html',
    'get_progress_bar_html',
    'get_alert_box_html',
]
