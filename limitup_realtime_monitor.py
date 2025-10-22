"""
涨停板预测系统 - 实时监控系统
支持10秒级刷新，WebSocket实时推送
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Any
from threading import Thread
import pandas as pd
import numpy as np

# Flask Web服务
try:
    from flask import Flask, render_template_string, jsonify
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask未安装，监控系统将在控制台模式运行")


class RealtimeMonitor:
    """实时监控器"""
    
    def __init__(
        self,
        refresh_interval: int = 10,
        enable_web: bool = True,
        port: int = 5000
    ):
        """
        初始化监控器
        
        Args:
            refresh_interval: 刷新间隔（秒）
            enable_web: 是否启用Web界面
            port: Web服务端口
        """
        self.refresh_interval = refresh_interval
        self.enable_web = enable_web and FLASK_AVAILABLE
        self.port = port
        
        # 监控数据
        self.metrics = {
            'prediction_count': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'limit_up_detected': 0,
            'last_update': None
        }
        
        self.history = []
        self.running = False
        
        # Web服务
        if self.enable_web:
            self.app = Flask(__name__)
            self.app.config['SECRET_KEY'] = 'limitup_monitor_secret'
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self._setup_routes()
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """更新监控指标"""
        self.metrics.update(metrics)
        self.metrics['last_update'] = datetime.now().isoformat()
        
        # 保存历史
        self.history.append({
            'timestamp': self.metrics['last_update'],
            **metrics
        })
        
        # 限制历史记录数量
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # 推送更新
        if self.enable_web:
            self.socketio.emit('metrics_update', self.metrics)
    
    def _setup_routes(self):
        """设置Web路由"""
        
        @self.app.route('/')
        def index():
            """监控Dashboard"""
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """获取当前指标"""
            return jsonify(self.metrics)
        
        @self.app.route('/api/history')
        def get_history():
            """获取历史数据"""
            return jsonify(self.history[-100:])
        
        @self.socketio.on('connect')
        def handle_connect():
            """WebSocket连接"""
            emit('metrics_update', self.metrics)
    
    def start(self):
        """启动监控服务"""
        self.running = True
        
        if self.enable_web:
            print(f"\n🌐 Web监控服务启动: http://localhost:{self.port}")
            print(f"刷新间隔: {self.refresh_interval}秒")
            
            # 在独立线程中启动模拟数据生成
            Thread(target=self._simulate_metrics, daemon=True).start()
            
            # 启动Web服务
            self.socketio.run(self.app, port=self.port, debug=False)
        else:
            print("\n📊 控制台监控模式启动...")
            self._console_monitor()
    
    def _simulate_metrics(self):
        """模拟实时指标更新（用于演示）"""
        while self.running:
            # 生成模拟指标
            metrics = {
                'prediction_count': self.metrics['prediction_count'] + np.random.randint(1, 10),
                'accuracy': min(0.99, self.metrics['accuracy'] + np.random.uniform(-0.02, 0.03)),
                'precision': min(0.99, max(0.60, np.random.uniform(0.75, 0.95))),
                'recall': min(0.99, max(0.60, np.random.uniform(0.70, 0.90))),
                'f1_score': min(0.99, max(0.60, np.random.uniform(0.72, 0.92))),
                'limit_up_detected': self.metrics['limit_up_detected'] + np.random.randint(0, 3)
            }
            
            self.update_metrics(metrics)
            time.sleep(self.refresh_interval)
    
    def _console_monitor(self):
        """控制台监控模式"""
        while self.running:
            print(f"\n{'='*60}")
            print(f"📊 涨停板预测实时监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"预测次数: {self.metrics['prediction_count']}")
            print(f"准确率: {self.metrics['accuracy']:.2%}")
            print(f"精确率: {self.metrics['precision']:.2%}")
            print(f"召回率: {self.metrics['recall']:.2%}")
            print(f"F1分数: {self.metrics['f1_score']:.2%}")
            print(f"检测涨停: {self.metrics['limit_up_detected']}")
            print(f"{'='*60}\n")
            
            time.sleep(self.refresh_interval)
    
    def stop(self):
        """停止监控"""
        self.running = False


# HTML Dashboard模板
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>涨停板预测实时监控</title>
    <meta charset="utf-8">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.3.0/dist/chart.umd.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                   gap: 15px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 32px; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; font-size: 14px; margin-top: 5px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; 
                          box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 400px; }
        .status { padding: 5px 10px; border-radius: 4px; display: inline-block; }
        .status.online { background: #4caf50; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 涨停板预测实时监控系统</h1>
        <p>实时更新间隔: 10秒 | <span class="status online">● 在线</span></p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value" id="prediction_count">0</div>
            <div class="metric-label">预测次数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="accuracy">0%</div>
            <div class="metric-label">准确率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="precision">0%</div>
            <div class="metric-label">精确率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="recall">0%</div>
            <div class="metric-label">召回率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="f1_score">0%</div>
            <div class="metric-label">F1分数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="limit_up_detected">0</div>
            <div class="metric-label">检测涨停</div>
        </div>
    </div>
    
    <div class="chart-container">
        <canvas id="metricsChart"></canvas>
    </div>
    
    <script>
        const socket = io();
        
        // 历史数据
        const history = {
            timestamps: [],
            accuracy: [],
            precision: [],
            recall: [],
            f1: []
        };
        
        // 初始化图表
        const ctx = document.getElementById('metricsChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: history.timestamps,
                datasets: [
                    {
                        label: '准确率',
                        data: history.accuracy,
                        borderColor: '#667eea',
                        tension: 0.4
                    },
                    {
                        label: 'F1分数',
                        data: history.f1,
                        borderColor: '#764ba2',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        // 监听指标更新
        socket.on('metrics_update', (data) => {
            // 更新数值
            document.getElementById('prediction_count').textContent = data.prediction_count;
            document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('precision').textContent = (data.precision * 100).toFixed(2) + '%';
            document.getElementById('recall').textContent = (data.recall * 100).toFixed(2) + '%';
            document.getElementById('f1_score').textContent = (data.f1_score * 100).toFixed(2) + '%';
            document.getElementById('limit_up_detected').textContent = data.limit_up_detected;
            
            // 更新图表
            const timestamp = new Date().toLocaleTimeString();
            history.timestamps.push(timestamp);
            history.accuracy.push(data.accuracy);
            history.precision.push(data.precision);
            history.recall.push(data.recall);
            history.f1.push(data.f1_score);
            
            // 限制历史长度
            if (history.timestamps.length > 30) {
                history.timestamps.shift();
                history.accuracy.shift();
                history.precision.shift();
                history.recall.shift();
                history.f1.shift();
            }
            
            chart.update();
        });
    </script>
</body>
</html>
"""


if __name__ == '__main__':
    print("="*60)
    print("涨停板预测系统 - 实时监控系统")
    print("="*60)
    
    monitor = RealtimeMonitor(
        refresh_interval=10,
        enable_web=True,
        port=5000
    )
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n\n⏹️  监控服务已停止")
        monitor.stop()
