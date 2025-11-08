"""
多渠道消息推送系统
支持企业微信、钉钉、邮件等多种推送渠道
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import logging
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.config_manager import get_config
except Exception as e:
    logging.warning(f"配置管理器导入失败: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """消息级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class NotificationChannel(Enum):
    """推送渠道"""
    WECHAT = "wechat"
    DINGTALK = "dingtalk"
    EMAIL = "email"


class Notifier:
    """
    统一消息推送器
    
    功能：
    1. 企业微信机器人推送
    2. 钉钉机器人推送
    3. 邮件推送
    4. 批量推送
    5. 推送历史记录
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化推送器
        
        Parameters:
        -----------
        config: Dict
            配置字典
        """
        if config is None:
            config_manager = get_config()
            self.config = config_manager.get_section('notification')
        else:
            self.config = config
        
        # 启用状态
        self.enabled = self.config.get('enable_notification', True)
        
        # 渠道配置
        self.channels = self.config.get('channels', [])
        self.wechat_webhook = self.config.get('wechat_webhook', '')
        self.dingtalk_webhook = self.config.get('dingtalk_webhook', '')
        
        # 邮件配置
        self.email_config = {
            'smtp_server': self.config.get('email_smtp_server', ''),
            'smtp_port': self.config.get('email_smtp_port', 587),
            'from': self.config.get('email_from', ''),
            'to': self.config.get('email_to', []),
            'password': self.config.get('email_password', '')
        }
        
        # 推送历史
        self.history = []
        
        logger.info(f"推送器初始化完成 - 启用: {self.enabled}, 渠道: {self.channels}")
    
    def send(
        self,
        title: str,
        content: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: Optional[List[str]] = None
    ) -> Dict:
        """
        发送消息
        
        Parameters:
        -----------
        title: str
            消息标题
        content: str
            消息内容
        level: NotificationLevel
            消息级别
        channels: List[str]
            推送渠道，如果为None则使用配置的所有渠道
            
        Returns:
        --------
        Dict: 推送结果
        """
        if not self.enabled:
            logger.info("消息推送未启用")
            return {'status': 'disabled', 'results': []}
        
        # 确定推送渠道
        target_channels = channels or self.channels
        if not target_channels:
            logger.warning("未配置推送渠道")
            return {'status': 'no_channels', 'results': []}
        
        # 记录推送请求
        timestamp = datetime.now()
        
        # 执行推送
        results = []
        for channel in target_channels:
            try:
                if channel == 'wechat':
                    result = self._send_wechat(title, content, level)
                elif channel == 'dingtalk':
                    result = self._send_dingtalk(title, content, level)
                elif channel == 'email':
                    result = self._send_email(title, content, level)
                else:
                    result = {'channel': channel, 'status': 'unsupported'}
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"推送失败 ({channel}): {e}")
                results.append({
                    'channel': channel,
                    'status': 'error',
                    'error': str(e)
                })
        
        # 记录历史
        self.history.append({
            'timestamp': timestamp,
            'title': title,
            'content': content,
            'level': level.value,
            'channels': target_channels,
            'results': results
        })
        
        return {
            'status': 'sent',
            'timestamp': timestamp,
            'results': results
        }
    
    def _send_wechat(
        self,
        title: str,
        content: str,
        level: NotificationLevel
    ) -> Dict:
        """
        企业微信推送
        
        Parameters:
        -----------
        title: str
            消息标题
        content: str
            消息内容
        level: NotificationLevel
            消息级别
        """
        if not self.wechat_webhook:
            return {'channel': 'wechat', 'status': 'not_configured'}
        
        # 构建消息
        emoji_map = {
            NotificationLevel.INFO: 'ℹ️',
            NotificationLevel.WARNING: '⚠️',
            NotificationLevel.ERROR: '❌',
            NotificationLevel.SUCCESS: '✅'
        }
        
        emoji = emoji_map.get(level, 'ℹ️')
        
        message = {
            "msgtype": "markdown",
            "markdown": {
                "content": f"## {emoji} {title}\n\n{content}"
            }
        }
        
        try:
            response = requests.post(
                self.wechat_webhook,
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"✓ 企业微信推送成功: {title}")
                    return {'channel': 'wechat', 'status': 'success'}
                else:
                    logger.error(f"企业微信推送失败: {result}")
                    return {
                        'channel': 'wechat',
                        'status': 'failed',
                        'error': result.get('errmsg', 'Unknown error')
                    }
            else:
                return {
                    'channel': 'wechat',
                    'status': 'failed',
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"企业微信推送异常: {e}")
            return {'channel': 'wechat', 'status': 'error', 'error': str(e)}
    
    def _send_dingtalk(
        self,
        title: str,
        content: str,
        level: NotificationLevel
    ) -> Dict:
        """
        钉钉推送
        
        Parameters:
        -----------
        title: str
            消息标题
        content: str
            消息内容
        level: NotificationLevel
            消息级别
        """
        if not self.dingtalk_webhook:
            return {'channel': 'dingtalk', 'status': 'not_configured'}
        
        # 构建消息
        emoji_map = {
            NotificationLevel.INFO: 'ℹ️',
            NotificationLevel.WARNING: '⚠️',
            NotificationLevel.ERROR: '❌',
            NotificationLevel.SUCCESS: '✅'
        }
        
        emoji = emoji_map.get(level, 'ℹ️')
        
        message = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": f"## {emoji} {title}\n\n{content}"
            }
        }
        
        try:
            response = requests.post(
                self.dingtalk_webhook,
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"✓ 钉钉推送成功: {title}")
                    return {'channel': 'dingtalk', 'status': 'success'}
                else:
                    logger.error(f"钉钉推送失败: {result}")
                    return {
                        'channel': 'dingtalk',
                        'status': 'failed',
                        'error': result.get('errmsg', 'Unknown error')
                    }
            else:
                return {
                    'channel': 'dingtalk',
                    'status': 'failed',
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"钉钉推送异常: {e}")
            return {'channel': 'dingtalk', 'status': 'error', 'error': str(e)}
    
    def _send_email(
        self,
        title: str,
        content: str,
        level: NotificationLevel
    ) -> Dict:
        """
        邮件推送
        
        Parameters:
        -----------
        title: str
            消息标题
        content: str
            消息内容
        level: NotificationLevel
            消息级别
        """
        if not all([
            self.email_config['smtp_server'],
            self.email_config['from'],
            self.email_config['to']
        ]):
            return {'channel': 'email', 'status': 'not_configured'}
        
        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[Qilin] {title}"
            msg['From'] = self.email_config['from']
            msg['To'] = ', '.join(self.email_config['to'])
            
            # HTML内容
            html_content = self._format_email_html(title, content, level)
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # 发送邮件
            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                
                # 如果配置了密码，则登录
                if self.email_config.get('password'):
                    server.login(
                        self.email_config['from'],
                        self.email_config['password']
                    )
                
                server.send_message(msg)
            
            logger.info(f"✓ 邮件推送成功: {title}")
            return {'channel': 'email', 'status': 'success'}
            
        except Exception as e:
            logger.error(f"邮件推送异常: {e}")
            return {'channel': 'email', 'status': 'error', 'error': str(e)}
    
    def _format_email_html(
        self,
        title: str,
        content: str,
        level: NotificationLevel
    ) -> str:
        """格式化邮件HTML"""
        color_map = {
            NotificationLevel.INFO: '#1890ff',
            NotificationLevel.WARNING: '#faad14',
            NotificationLevel.ERROR: '#f5222d',
            NotificationLevel.SUCCESS: '#52c41a'
        }
        
        color = color_map.get(level, '#1890ff')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f5f5f5; padding: 20px; border-radius: 0 0 5px 5px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #999; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{title}</h2>
                </div>
                <div class="content">
                    <pre style="white-space: pre-wrap; word-wrap: break-word;">{content}</pre>
                </div>
                <div class="footer">
                    <p>Qilin量化交易系统 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    # ==================== 便捷方法 ====================
    
    def send_auction_signal(self, signals: List[Dict]) -> Dict:
        """
        发送竞价信号
        
        Parameters:
        -----------
        signals: List[Dict]
            竞价信号列表
        """
        if not signals:
            return {'status': 'no_signals'}
        
        # 构建消息内容
        content = "### 竞价买入信号\n\n"
        content += f"信号数量: {len(signals)}\n\n"
        
        for i, signal in enumerate(signals[:10], 1):  # 最多显示10个
            content += f"{i}. **{signal.get('symbol', 'N/A')}** {signal.get('name', 'N/A')}\n"
            content += f"   - 竞价强度: {signal.get('auction_strength', 0):.2%}\n"
            content += f"   - 竞价价格: {signal.get('auction_price', 0):.2f}\n\n"
        
        if len(signals) > 10:
            content += f"\n... 还有 {len(signals) - 10} 个信号未显示"
        
        return self.send(
            title="竞价买入信号",
            content=content,
            level=NotificationLevel.SUCCESS
        )
    
    def send_buy_notification(self, orders: List[Dict]) -> Dict:
        """
        发送买入通知
        
        Parameters:
        -----------
        orders: List[Dict]
            买入订单列表
        """
        if not orders:
            return {'status': 'no_orders'}
        
        # 构建消息内容
        total_amount = sum(o.get('amount', 0) for o in orders)
        
        content = "### 买入执行通知\n\n"
        content += f"订单数量: {len(orders)}\n"
        content += f"总金额: ¥{total_amount:,.2f}\n\n"
        
        for i, order in enumerate(orders[:10], 1):
            content += f"{i}. **{order.get('symbol', 'N/A')}**\n"
            content += f"   - 价格: {order.get('price', 0):.2f}\n"
            content += f"   - 数量: {order.get('volume', 0)}\n"
            content += f"   - 金额: ¥{order.get('amount', 0):,.2f}\n\n"
        
        if len(orders) > 10:
            content += f"\n... 还有 {len(orders) - 10} 笔订单未显示"
        
        return self.send(
            title="买入执行通知",
            content=content,
            level=NotificationLevel.INFO
        )
    
    def send_sell_notification(self, orders: List[Dict]) -> Dict:
        """
        发送卖出通知
        
        Parameters:
        -----------
        orders: List[Dict]
            卖出订单列表
        """
        if not orders:
            return {'status': 'no_orders'}
        
        # 构建消息内容
        total_profit = sum(o.get('profit', 0) for o in orders)
        
        content = "### 卖出执行通知\n\n"
        content += f"订单数量: {len(orders)}\n"
        content += f"总盈亏: ¥{total_profit:,.2f}\n\n"
        
        for i, order in enumerate(orders[:10], 1):
            profit = order.get('profit', 0)
            profit_rate = order.get('profit_rate', 0)
            
            content += f"{i}. **{order.get('symbol', 'N/A')}**\n"
            content += f"   - 卖出价格: {order.get('sell_price', 0):.2f}\n"
            content += f"   - 盈亏: ¥{profit:,.2f} ({profit_rate:+.2%})\n\n"
        
        if len(orders) > 10:
            content += f"\n... 还有 {len(orders) - 10} 笔订单未显示"
        
        level = NotificationLevel.SUCCESS if total_profit > 0 else NotificationLevel.WARNING
        
        return self.send(
            title="卖出执行通知",
            content=content,
            level=level
        )
    
    def send_daily_report(self, report: Dict) -> Dict:
        """
        发送每日报告
        
        Parameters:
        -----------
        report: Dict
            日度报告
        """
        content = "### 每日交易报告\n\n"
        content += f"日期: {report.get('date', 'N/A')}\n\n"
        content += f"候选筛选: {report.get('candidates', 0)} 只\n"
        content += f"买入订单: {report.get('buy_orders', 0)} 笔\n"
        content += f"卖出订单: {report.get('sell_orders', 0)} 笔\n\n"
        content += f"当日盈亏: ¥{report.get('profit', 0):,.2f}\n"
        content += f"盈亏比例: {report.get('profit_rate', 0):+.2%}\n"
        
        level = NotificationLevel.SUCCESS if report.get('profit', 0) > 0 else NotificationLevel.INFO
        
        return self.send(
            title="每日交易报告",
            content=content,
            level=level
        )
    
    def send_error_alert(self, error_message: str) -> Dict:
        """
        发送错误告警
        
        Parameters:
        -----------
        error_message: str
            错误信息
        """
        return self.send(
            title="系统错误告警",
            content=f"```\n{error_message}\n```",
            level=NotificationLevel.ERROR
        )
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """
        获取推送历史
        
        Parameters:
        -----------
        limit: int
            返回最近N条记录
        """
        return self.history[-limit:]


# 全局推送器实例
_global_notifier: Optional[Notifier] = None


def get_notifier(config: Optional[Dict] = None) -> Notifier:
    """
    获取全局推送器实例（单例模式）
    
    Parameters:
    -----------
    config: Dict
        配置字典
        
    Returns:
    --------
    Notifier: 推送器实例
    """
    global _global_notifier
    
    if _global_notifier is None:
        _global_notifier = Notifier(config)
    
    return _global_notifier


# 使用示例
if __name__ == "__main__":
    print("\n" + "="*80)
    print("消息推送系统测试")
    print("="*80)
    
    # 创建推送器（使用测试配置）
    test_config = {
        'enable_notification': True,
        'channels': [],  # 留空，避免实际推送
        'wechat_webhook': '',
        'dingtalk_webhook': '',
        'email_smtp_server': '',
        'email_from': '',
        'email_to': []
    }
    
    notifier = Notifier(config=test_config)
    
    # 测试1：基本推送
    print("\n测试1: 基本推送")
    result = notifier.send(
        title="测试消息",
        content="这是一条测试消息",
        level=NotificationLevel.INFO
    )
    print(f"推送结果: {result['status']}")
    
    # 测试2：竞价信号推送
    print("\n测试2: 竞价信号推送")
    signals = [
        {'symbol': '000001.SZ', 'name': '平安银行', 'auction_strength': 0.85, 'auction_price': 12.50},
        {'symbol': '600519.SH', 'name': '贵州茅台', 'auction_strength': 0.92, 'auction_price': 1680.00}
    ]
    result = notifier.send_auction_signal(signals)
    print(f"推送结果: {result['status']}")
    
    # 测试3: 买入通知
    print("\n测试3: 买入通知")
    orders = [
        {'symbol': '000001.SZ', 'price': 12.50, 'volume': 1000, 'amount': 12500},
        {'symbol': '600519.SH', 'price': 1680.00, 'volume': 100, 'amount': 168000}
    ]
    result = notifier.send_buy_notification(orders)
    print(f"推送结果: {result['status']}")
    
    # 测试4: 卖出通知
    print("\n测试4: 卖出通知")
    orders = [
        {'symbol': '000001.SZ', 'sell_price': 13.00, 'profit': 500, 'profit_rate': 0.04},
        {'symbol': '600519.SH', 'sell_price': 1750.00, 'profit': 7000, 'profit_rate': 0.042}
    ]
    result = notifier.send_sell_notification(orders)
    print(f"推送结果: {result['status']}")
    
    # 测试5: 每日报告
    print("\n测试5: 每日报告")
    report = {
        'date': '2024-11-01',
        'candidates': 23,
        'buy_orders': 12,
        'sell_orders': 8,
        'profit': 3240.50,
        'profit_rate': 0.0254
    }
    result = notifier.send_daily_report(report)
    print(f"推送结果: {result['status']}")
    
    # 查看推送历史
    print("\n推送历史:")
    for record in notifier.get_history():
        print(f"  {record['timestamp']}: {record['title']} - {record['level']}")
    
    print("\n" + "="*80)
    print("✅ 消息推送系统测试完成！")
    print("="*80)
