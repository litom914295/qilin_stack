"""
可视化报告生成器 (Visual Report Generator)
生成图表丰富的HTML格式回测报告

核心功能：
1. 权益曲线图
2. 回撤曲线图
3. 月度收益热力图
4. 交易分布图
5. 风险收益散点图
6. 完整HTML报告
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class ReportData:
    """报告数据"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    
    # 指标
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    
    # 时序数据
    equity_curve: pd.Series
    drawdowns: pd.Series
    returns: pd.Series
    
    # 交易记录
    trades: Optional[List[Dict]] = None


class ReportGenerator:
    """可视化报告生成器"""
    
    def __init__(self):
        self.reports: List[ReportData] = []
    
    def add_report(self, data: ReportData):
        """添加报告数据"""
        self.reports.append(data)
        print(f"✅ 添加报告: {data.strategy_name}")
    
    def generate_html(self, output_path: str = "backtest_report.html"):
        """
        生成HTML报告
        
        Args:
            output_path: 输出文件路径
        """
        html_content = self._build_html()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ 报告已生成: {output_path}")
        return output_path
    
    def _build_html(self) -> str:
        """构建HTML内容"""
        # HTML模板
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>麒麟量化 - 回测报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 18px;
            opacity: 0.9;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: #333;
        }}
        
        .metric-value.positive {{
            color: #10b981;
        }}
        
        .metric-value.negative {{
            color: #ef4444;
        }}
        
        .chart-section {{
            padding: 40px;
        }}
        
        .chart-title {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 40px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            font-size: 14px;
        }}
        
        .strategy-tabs {{
            display: flex;
            gap: 10px;
            padding: 20px 40px;
            background: #f8f9fa;
            overflow-x: auto;
        }}
        
        .strategy-tab {{
            padding: 12px 24px;
            background: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        
        .strategy-tab:hover {{
            background: #667eea;
            color: white;
        }}
        
        .strategy-tab.active {{
            background: #667eea;
            color: white;
        }}
        
        .strategy-content {{
            display: none;
        }}
        
        .strategy-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦄 麒麟量化回测报告</h1>
            <p>Qilin Stack Backtest Report</p>
        </div>
        
        {self._build_strategy_tabs()}
        
        {self._build_all_strategies()}
        
        <div class="footer">
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Powered by Qilin Stack Quantitative Trading System</p>
        </div>
    </div>
    
    <script>
        // 策略切换
        function switchStrategy(index) {{
            // 隐藏所有内容
            document.querySelectorAll('.strategy-content').forEach(el => {{
                el.classList.remove('active');
            }});
            document.querySelectorAll('.strategy-tab').forEach(el => {{
                el.classList.remove('active');
            }});
            
            // 显示选中的内容
            document.getElementById('strategy-' + index).classList.add('active');
            document.querySelectorAll('.strategy-tab')[index].classList.add('active');
        }}
        
        // 默认显示第一个策略
        if (document.querySelectorAll('.strategy-tab').length > 0) {{
            switchStrategy(0);
        }}
    </script>
</body>
</html>
"""
        return html
    
    def _build_strategy_tabs(self) -> str:
        """构建策略标签页"""
        if len(self.reports) == 0:
            return ""
        
        tabs_html = '<div class="strategy-tabs">\n'
        
        for i, report in enumerate(self.reports):
            tabs_html += f'    <button class="strategy-tab" onclick="switchStrategy({i})">{report.strategy_name}</button>\n'
        
        tabs_html += '</div>\n'
        
        return tabs_html
    
    def _build_all_strategies(self) -> str:
        """构建所有策略内容"""
        if len(self.reports) == 0:
            return "<p style='text-align:center; padding:40px;'>暂无报告数据</p>"
        
        all_html = ""
        
        for i, report in enumerate(self.reports):
            all_html += f'<div id="strategy-{i}" class="strategy-content">\n'
            all_html += self._build_strategy_content(report, i)
            all_html += '</div>\n'
        
        return all_html
    
    def _build_strategy_content(self, report: ReportData, index: int) -> str:
        """构建单个策略内容"""
        content = ""
        
        # 关键指标卡片
        content += self._build_metrics_cards(report)
        
        # 图表区域
        content += '<div class="chart-section">\n'
        
        # 1. 权益曲线
        content += self._build_equity_chart(report, index)
        
        # 2. 回撤曲线
        content += self._build_drawdown_chart(report, index)
        
        # 3. 月度收益热力图
        content += self._build_monthly_heatmap(report, index)
        
        # 4. 收益分布直方图
        content += self._build_returns_distribution(report, index)
        
        content += '</div>\n'
        
        return content
    
    def _build_metrics_cards(self, report: ReportData) -> str:
        """构建指标卡片"""
        metrics = [
            {
                "label": "总收益率",
                "value": f"{report.total_return:.2%}",
                "class": "positive" if report.total_return > 0 else "negative"
            },
            {
                "label": "年化收益率",
                "value": f"{report.annual_return:.2%}",
                "class": "positive" if report.annual_return > 0 else "negative"
            },
            {
                "label": "夏普比率",
                "value": f"{report.sharpe_ratio:.2f}",
                "class": "positive" if report.sharpe_ratio > 0 else "negative"
            },
            {
                "label": "最大回撤",
                "value": f"{report.max_drawdown:.2%}",
                "class": "negative"
            },
            {
                "label": "胜率",
                "value": f"{report.win_rate:.2%}",
                "class": "positive" if report.win_rate > 0.5 else ""
            },
            {
                "label": "交易次数",
                "value": str(report.total_trades),
                "class": ""
            }
        ]
        
        html = '<div class="metrics">\n'
        
        for metric in metrics:
            html += f"""
    <div class="metric-card">
        <div class="metric-label">{metric['label']}</div>
        <div class="metric-value {metric['class']}">{metric['value']}</div>
    </div>
"""
        
        html += '</div>\n'
        
        return html
    
    def _build_equity_chart(self, report: ReportData, index: int) -> str:
        """构建权益曲线图"""
        equity_data = report.equity_curve.reset_index()
        dates = equity_data.iloc[:, 0].dt.strftime('%Y-%m-%d').tolist()
        values = equity_data.iloc[:, 1].tolist()
        
        # 使用Plotly生成图表
        chart_id = f"equity-chart-{index}"
        
        html = f"""
<div class="chart-container">
    <div class="chart-title">📈 权益曲线</div>
    <div id="{chart_id}"></div>
    <script>
        var trace = {{
            x: {dates},
            y: {values},
            type: 'scatter',
            mode: 'lines',
            name: '权益',
            line: {{
                color: '#667eea',
                width: 2
            }},
            fill: 'tozeroy',
            fillcolor: 'rgba(102, 126, 234, 0.1)'
        }};
        
        var layout = {{
            title: '',
            xaxis: {{
                title: '日期',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }},
            yaxis: {{
                title: '权益',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }},
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 20, b: 60, l: 60, r: 20 }}
        }};
        
        Plotly.newPlot('{chart_id}', [trace], layout, {{responsive: true}});
    </script>
</div>
"""
        
        return html
    
    def _build_drawdown_chart(self, report: ReportData, index: int) -> str:
        """构建回撤曲线图"""
        dd_data = report.drawdowns.reset_index()
        dates = dd_data.iloc[:, 0].dt.strftime('%Y-%m-%d').tolist()
        values = (dd_data.iloc[:, 1] * 100).tolist()  # 转换为百分比
        
        chart_id = f"drawdown-chart-{index}"
        
        html = f"""
<div class="chart-container">
    <div class="chart-title">📉 回撤曲线</div>
    <div id="{chart_id}"></div>
    <script>
        var trace = {{
            x: {dates},
            y: {values},
            type: 'scatter',
            mode: 'lines',
            name: '回撤',
            line: {{
                color: '#ef4444',
                width: 2
            }},
            fill: 'tozeroy',
            fillcolor: 'rgba(239, 68, 68, 0.1)'
        }};
        
        var layout = {{
            title: '',
            xaxis: {{
                title: '日期',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }},
            yaxis: {{
                title: '回撤 (%)',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }},
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 20, b: 60, l: 60, r: 20 }}
        }};
        
        Plotly.newPlot('{chart_id}', [trace], layout, {{responsive: true}});
    </script>
</div>
"""
        
        return html
    
    def _build_monthly_heatmap(self, report: ReportData, index: int) -> str:
        """构建月度收益热力图"""
        # 计算月度收益
        returns = report.returns.copy()
        returns.index = pd.to_datetime(returns.index)
        
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum()
        monthly_returns = (monthly_returns * 100).round(2)  # 转换为百分比
        
        # 重塑为矩阵
        years = sorted(set([idx[0] for idx in monthly_returns.index]))
        months = list(range(1, 13))
        
        # 构建数据矩阵
        z_data = []
        for year in years:
            row = []
            for month in months:
                if (year, month) in monthly_returns.index:
                    row.append(monthly_returns[(year, month)])
                else:
                    row.append(None)
            z_data.append(row)
        
        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月', 
                       '7月', '8月', '9月', '10月', '11月', '12月']
        
        chart_id = f"heatmap-chart-{index}"
        
        html = f"""
<div class="chart-container">
    <div class="chart-title">🔥 月度收益热力图</div>
    <div id="{chart_id}"></div>
    <script>
        var data = [{{
            z: {z_data},
            x: {month_labels},
            y: {years},
            type: 'heatmap',
            colorscale: [
                [0, '#ef4444'],
                [0.5, '#ffffff'],
                [1, '#10b981']
            ],
            zmid: 0,
            text: {z_data},
            hovertemplate: '%{{y}}年 %{{x}}: %{{z:.2f}}%<extra></extra>',
            colorbar: {{
                title: '收益率 (%)'
            }}
        }}];
        
        var layout = {{
            title: '',
            xaxis: {{
                title: '月份',
                side: 'bottom'
            }},
            yaxis: {{
                title: '年份'
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 20, b: 60, l: 60, r: 80 }}
        }};
        
        Plotly.newPlot('{chart_id}', data, layout, {{responsive: true}});
    </script>
</div>
"""
        
        return html
    
    def _build_returns_distribution(self, report: ReportData, index: int) -> str:
        """构建收益分布直方图"""
        returns = (report.returns * 100).dropna().tolist()  # 转换为百分比
        
        chart_id = f"distribution-chart-{index}"
        
        html = f"""
<div class="chart-container">
    <div class="chart-title">📊 日收益率分布</div>
    <div id="{chart_id}"></div>
    <script>
        var trace = {{
            x: {returns},
            type: 'histogram',
            marker: {{
                color: '#667eea',
                line: {{
                    color: '#4c5fd5',
                    width: 1
                }}
            }},
            opacity: 0.7,
            nbinsx: 50
        }};
        
        var layout = {{
            title: '',
            xaxis: {{
                title: '日收益率 (%)',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }},
            yaxis: {{
                title: '频次',
                showgrid: true,
                gridcolor: '#f0f0f0'
            }},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {{ t: 20, b: 60, l: 60, r: 20 }}
        }};
        
        Plotly.newPlot('{chart_id}', [trace], layout, {{responsive: true}});
    </script>
</div>
"""
        
        return html


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # 策略1
    returns_1 = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    equity_1 = (1 + returns_1).cumprod()
    
    def calculate_drawdown(equity: pd.Series) -> pd.Series:
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown
    
    drawdowns_1 = calculate_drawdown(equity_1)
    
    report1 = ReportData(
        strategy_name="保守型策略",
        start_date=dates[0],
        end_date=dates[-1],
        total_return=equity_1.iloc[-1] - 1,
        annual_return=(equity_1.iloc[-1] - 1),
        sharpe_ratio=returns_1.mean() / returns_1.std() * np.sqrt(252),
        max_drawdown=drawdowns_1.min(),
        win_rate=0.58,
        total_trades=150,
        equity_curve=equity_1,
        drawdowns=drawdowns_1,
        returns=returns_1
    )
    
    # 策略2
    returns_2 = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    equity_2 = (1 + returns_2).cumprod()
    drawdowns_2 = calculate_drawdown(equity_2)
    
    report2 = ReportData(
        strategy_name="激进型策略",
        start_date=dates[0],
        end_date=dates[-1],
        total_return=equity_2.iloc[-1] - 1,
        annual_return=(equity_2.iloc[-1] - 1),
        sharpe_ratio=returns_2.mean() / returns_2.std() * np.sqrt(252),
        max_drawdown=drawdowns_2.min(),
        win_rate=0.52,
        total_trades=300,
        equity_curve=equity_2,
        drawdowns=drawdowns_2,
        returns=returns_2
    )
    
    # 生成报告
    generator = ReportGenerator()
    generator.add_report(report1)
    generator.add_report(report2)
    
    output_file = generator.generate_html("qilin_backtest_report.html")
    
    print(f"\n📊 报告预览:")
    print(f"  - 策略数量: {len(generator.reports)}")
    print(f"  - 报告文件: {output_file}")
    print(f"  - 可在浏览器中打开查看完整报告")
    
    print("\n✅ 完成")
