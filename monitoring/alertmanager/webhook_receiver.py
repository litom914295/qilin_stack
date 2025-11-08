"""
Alertmanager Webhook接收服务
处理告警并发送到多个渠道（企业微信、钉钉、短信等）
"""

from flask import Flask, request, jsonify
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class AlertWebhookReceiver:
    """告警Webhook接收器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化接收器
        
        Parameters:
        -----------
        config: Dict
            配置信息，包含企业微信、钉钉、短信等配置
        """
        self.config = config or {}
    
    def process_alerts(self, alerts_data: Dict) -> Dict:
        """
        处理告警数据
        
        Parameters:
        -----------
        alerts_data: Dict
            来自Alertmanager的告警数据
            
        Returns:
        --------
        Dict: 处理结果
        """
        alerts = alerts_data.get('alerts', [])
        
        logger.info(f"收到 {len(alerts)} 条告警")
        
        results = []
        for alert in alerts:
            result = self._process_single_alert(alert)
            results.append(result)
        
        return {
            'status': 'success',
            'processed': len(results),
            'results': results
        }
    
    def _process_single_alert(self, alert: Dict) -> Dict:
        """处理单个告警"""
        # 提取告警信息
        labels = alert.get('labels', {})
        annotations = alert.get('annotations', {})
        status = alert.get('status', 'unknown')
        
        alertname = labels.get('alertname', 'Unknown')
        severity = labels.get('severity', 'unknown')
        category = labels.get('category', 'unknown')
        
        summary = annotations.get('summary', '')
        description = annotations.get('description', '')
        
        logger.info(f"处理告警: {alertname} ({severity}) - {status}")
        
        # 根据严重程度和类别决定通知渠道
        result = {
            'alertname': alertname,
            'severity': severity,
            'category': category,
            'status': status,
            'notifications': []
        }
        
        # 交易系统告警 - 多渠道通知
        if category == 'trading':
            result['notifications'].extend([
                self._send_wechat(alert),
                self._send_dingtalk(alert),
                self._send_sms(alert) if severity == 'critical' else None
            ])
        
        # 严重告警 - 短信通知
        elif severity == 'critical':
            result['notifications'].extend([
                self._send_wechat(alert),
                self._send_sms(alert)
            ])
        
        # 其他告警 - 企业微信
        else:
            result['notifications'].append(self._send_wechat(alert))
        
        # 过滤None
        result['notifications'] = [n for n in result['notifications'] if n]
        
        return result
    
    def _send_wechat(self, alert: Dict) -> Dict:
        """发送企业微信通知"""
        try:
            webhook_url = self.config.get('wechat_webhook_url')
            if not webhook_url:
                return {'channel': 'wechat', 'status': 'skipped', 'reason': 'no webhook url'}
            
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            status = alert.get('status', 'unknown')
            
            # 构造消息
            color = 'warning' if status == 'firing' else 'info'
            title = f"🚨 {labels.get('alertname', 'Unknown')}"
            
            content = f"""
**告警级别**: {labels.get('severity', 'unknown')}
**告警分类**: {labels.get('category', 'unknown')}
**告警状态**: {status}
**服务**: {labels.get('service', 'N/A')}
**实例**: {labels.get('instance', 'N/A')}
            
**摘要**: {annotations.get('summary', '')}
**详情**: {annotations.get('description', '')}
            
**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": f"# {title}\n{content}"
                }
            }
            
            response = requests.post(webhook_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"企业微信通知发送成功: {labels.get('alertname')}")
                return {'channel': 'wechat', 'status': 'success'}
            else:
                logger.error(f"企业微信通知发送失败: {response.text}")
                return {'channel': 'wechat', 'status': 'failed', 'error': response.text}
        
        except Exception as e:
            logger.error(f"企业微信通知异常: {str(e)}")
            return {'channel': 'wechat', 'status': 'error', 'error': str(e)}
    
    def _send_dingtalk(self, alert: Dict) -> Dict:
        """发送钉钉通知"""
        try:
            webhook_url = self.config.get('dingtalk_webhook_url')
            if not webhook_url:
                return {'channel': 'dingtalk', 'status': 'skipped', 'reason': 'no webhook url'}
            
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            status = alert.get('status', 'unknown')
            
            # 构造消息
            title = f"麒麟量化告警 - {labels.get('alertname', 'Unknown')}"
            text = f"""
### {title}
            
- **级别**: {labels.get('severity', 'unknown')}
- **分类**: {labels.get('category', 'unknown')}
- **状态**: {status}
- **服务**: {labels.get('service', 'N/A')}
            
**{annotations.get('summary', '')}**
            
{annotations.get('description', '')}
            
> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": text
                }
            }
            
            response = requests.post(webhook_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"钉钉通知发送成功: {labels.get('alertname')}")
                return {'channel': 'dingtalk', 'status': 'success'}
            else:
                logger.error(f"钉钉通知发送失败: {response.text}")
                return {'channel': 'dingtalk', 'status': 'failed', 'error': response.text}
        
        except Exception as e:
            logger.error(f"钉钉通知异常: {str(e)}")
            return {'channel': 'dingtalk', 'status': 'error', 'error': str(e)}
    
    def _send_sms(self, alert: Dict) -> Dict:
        """发送短信通知（严重告警）"""
        try:
            sms_api = self.config.get('sms_api_url')
            if not sms_api:
                return {'channel': 'sms', 'status': 'skipped', 'reason': 'no sms api'}
            
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            
            # 构造短信内容（精简）
            message = f"[麒麟量化严重告警] {labels.get('alertname')}: {annotations.get('summary', '')}"
            
            # 获取接收人列表
            recipients = self.config.get('sms_recipients', [])
            
            payload = {
                'recipients': recipients,
                'message': message
            }
            
            # 调用短信API
            response = requests.post(sms_api, json=payload, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"短信通知发送成功: {labels.get('alertname')}")
                return {'channel': 'sms', 'status': 'success', 'recipients': len(recipients)}
            else:
                logger.error(f"短信通知发送失败: {response.text}")
                return {'channel': 'sms', 'status': 'failed', 'error': response.text}
        
        except Exception as e:
            logger.error(f"短信通知异常: {str(e)}")
            return {'channel': 'sms', 'status': 'error', 'error': str(e)}


# 全局接收器实例
receiver = AlertWebhookReceiver(config={
    'wechat_webhook_url': 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY',
    'dingtalk_webhook_url': 'https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN',
    'sms_api_url': 'http://sms-service:8080/api/send',
    'sms_recipients': ['+86138****1234', '+86139****5678']
})


@app.route('/webhook/alerts', methods=['POST'])
def handle_alerts():
    """处理通用告警"""
    try:
        alerts_data = request.get_json()
        logger.info(f"收到告警webhook: {json.dumps(alerts_data, indent=2)}")
        
        result = receiver.process_alerts(alerts_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"处理告警失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/webhook/trading-alerts', methods=['POST'])
def handle_trading_alerts():
    """处理交易系统告警（高优先级）"""
    try:
        alerts_data = request.get_json()
        logger.warning(f"收到交易系统告警: {json.dumps(alerts_data, indent=2)}")
        
        # 交易系统告警立即处理
        result = receiver.process_alerts(alerts_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"处理交易告警失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/webhook/sms-alerts', methods=['POST'])
def handle_sms_alerts():
    """处理短信告警"""
    try:
        alerts_data = request.get_json()
        logger.critical(f"收到严重告警（短信）: {json.dumps(alerts_data, indent=2)}")
        
        result = receiver.process_alerts(alerts_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"处理短信告警失败: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    logger.info("启动告警Webhook接收服务...")
    app.run(host='0.0.0.0', port=5001, debug=False)
