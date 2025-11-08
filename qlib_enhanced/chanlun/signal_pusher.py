"""
信号推送系统 - Phase 5.3
功能: 邮件、微信、钉钉、Webhook多渠道推送
双模式复用: Qlib回测报告 + 独立系统实时信号
"""

import logging
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PushMessage:
    """推送消息"""
    title: str
    content: str
    level: str = "info"  # info|warning|error
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SignalPusher:
    """信号推送器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled_channels = config.get('enabled_channels', [])
    
    def push(self, message: PushMessage) -> Dict[str, bool]:
        """推送消息到所有渠道"""
        results = {}
        
        for channel in self.enabled_channels:
            try:
                if channel == 'email':
                    results['email'] = self._push_email(message)
                elif channel == 'wechat':
                    results['wechat'] = self._push_wechat(message)
                elif channel == 'dingtalk':
                    results['dingtalk'] = self._push_dingtalk(message)
                elif channel == 'webhook':
                    results['webhook'] = self._push_webhook(message)
            except Exception as e:
                logger.error(f"{channel}推送失败: {e}")
                results[channel] = False
        
        return results
    
    def _push_email(self, msg: PushMessage) -> bool:
        """邮件推送"""
        email_config = self.config.get('email', {})
        
        if not email_config.get('enabled', False):
            return False
        
        # 实际实现需要SMTP配置
        logger.info(f"[邮件] {msg.title}: {msg.content}")
        return True
    
    def _push_wechat(self, msg: PushMessage) -> bool:
        """微信推送(企业微信机器人)"""
        wechat_config = self.config.get('wechat', {})
        webhook_url = wechat_config.get('webhook_url')
        
        if not webhook_url:
            return False
        
        payload = {
            "msgtype": "text",
            "text": {
                "content": f"{msg.title}\n{msg.content}"
            }
        }
        
        try:
            resp = requests.post(webhook_url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"微信推送失败: {e}")
            return False
    
    def _push_dingtalk(self, msg: PushMessage) -> bool:
        """钉钉推送"""
        dingtalk_config = self.config.get('dingtalk', {})
        webhook_url = dingtalk_config.get('webhook_url')
        
        if not webhook_url:
            return False
        
        payload = {
            "msgtype": "text",
            "text": {
                "content": f"{msg.title}\n{msg.content}"
            }
        }
        
        try:
            resp = requests.post(webhook_url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"钉钉推送失败: {e}")
            return False
    
    def _push_webhook(self, msg: PushMessage) -> bool:
        """自定义Webhook推送"""
        webhook_config = self.config.get('webhook', {})
        url = webhook_config.get('url')
        
        if not url:
            return False
        
        payload = {
            "title": msg.title,
            "content": msg.content,
            "level": msg.level,
            "timestamp": msg.timestamp.isoformat()
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Webhook推送失败: {e}")
            return False


def push_chanlun_signal(signal: Dict, pusher: SignalPusher):
    """推送缠论信号"""
    if not signal:
        return
    
    title = f"缠论信号: {signal.get('symbol', 'unknown')}"
    content = f"类型: {signal.get('type', 'unknown')}\n"
    content += f"强度: {signal.get('strength', 0):.2f}\n"
    content += f"原因: {signal.get('reason', '')}"
    
    msg = PushMessage(title, content, level="info")
    pusher.push(msg)
