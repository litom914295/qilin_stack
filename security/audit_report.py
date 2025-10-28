"""
审计报告生成器（P0-5.3）
生成定期审计报告（日报/周报/月报）
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import json

from security.audit_enhanced import AuditLogger, AuditEventType

logger = logging.getLogger(__name__)


class AuditReportGenerator:
    """审计报告生成器"""
    
    def __init__(
        self,
        audit_logger: AuditLogger,
        output_dir: str = "reports/audit"
    ):
        """
        初始化报告生成器
        
        Args:
            audit_logger: 审计日志记录器
            output_dir: 输出目录
        """
        self.audit_logger = audit_logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Audit report generator initialized, output: {output_dir}")
    
    def generate_daily_report(
        self,
        date: Optional[datetime] = None
    ) -> Dict:
        """
        生成日报
        
        Args:
            date: 报告日期，默认为昨天
            
        Returns:
            报告数据字典
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        return self._generate_report(
            start_date=start_date,
            end_date=end_date,
            report_type="日报",
            period_name=date.strftime("%Y年%m月%d日")
        )
    
    def generate_weekly_report(
        self,
        week_start: Optional[datetime] = None
    ) -> Dict:
        """
        生成周报
        
        Args:
            week_start: 周开始日期，默认为上周一
            
        Returns:
            报告数据字典
        """
        if week_start is None:
            today = datetime.now()
            # 上周一
            week_start = today - timedelta(days=today.weekday() + 7)
        
        start_date = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=7)
        
        return self._generate_report(
            start_date=start_date,
            end_date=end_date,
            report_type="周报",
            period_name=f"{start_date.strftime('%Y年%m月%d日')} - {end_date.strftime('%m月%d日')}"
        )
    
    def generate_monthly_report(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Dict:
        """
        生成月报
        
        Args:
            year: 年份，默认为上个月
            month: 月份，默认为上个月
            
        Returns:
            报告数据字典
        """
        if year is None or month is None:
            today = datetime.now()
            if today.month == 1:
                year = today.year - 1
                month = 12
            else:
                year = today.year
                month = today.month - 1
        
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        return self._generate_report(
            start_date=start_date,
            end_date=end_date,
            report_type="月报",
            period_name=f"{year}年{month}月"
        )
    
    def _generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str,
        period_name: str
    ) -> Dict:
        """
        生成报告核心逻辑
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            report_type: 报告类型
            period_name: 时间段名称
            
        Returns:
            报告数据字典
        """
        # 查询审计事件
        events = self.audit_logger.query_events(
            start_date=start_date,
            end_date=end_date
        )
        
        # 统计分析
        stats = self._analyze_events(events)
        
        # 构建报告
        report = {
            "report_type": report_type,
            "period_name": period_name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "events": events
        }
        
        return report
    
    def _analyze_events(self, events: List[Dict]) -> Dict:
        """
        分析事件统计信息
        
        Args:
            events: 事件列表
            
        Returns:
            统计信息字典
        """
        if not events:
            return {
                "total_events": 0,
                "event_types": {},
                "users": {},
                "success_rate": 1.0,
                "pii_detected": 0,
                "pii_masked": 0,
                "top_users": [],
                "top_actions": [],
                "failed_events": []
            }
        
        # 基础统计
        total_events = len(events)
        
        # 事件类型统计
        event_types = Counter(e["event_type"] for e in events)
        
        # 用户统计
        users = Counter(e["user_id"] for e in events)
        
        # 成功率
        success_events = sum(1 for e in events if e["result"] == "success")
        success_rate = success_events / total_events if total_events > 0 else 0
        
        # PII统计
        pii_detected = sum(1 for e in events if e.get("pii_detected", False))
        pii_masked = sum(1 for e in events if e.get("pii_masked", False))
        
        # 失败事件
        failed_events = [
            e for e in events
            if e["result"] in ["failure", "error"]
        ]
        
        # 操作统计
        actions = Counter(e["action"] for e in events)
        
        stats = {
            "total_events": total_events,
            "event_types": dict(event_types.most_common()),
            "users": dict(users.most_common()),
            "success_rate": round(success_rate, 4),
            "pii_detected": pii_detected,
            "pii_masked": pii_masked,
            "pii_detection_rate": round(pii_detected / total_events, 4) if total_events > 0 else 0,
            "top_users": users.most_common(10),
            "top_actions": actions.most_common(10),
            "failed_events": failed_events[:20],  # 最多20个失败事件
            "failed_count": len(failed_events)
        }
        
        return stats
    
    def export_html(
        self,
        report: Dict,
        filename: Optional[str] = None
    ) -> str:
        """
        导出HTML报告
        
        Args:
            report: 报告数据
            filename: 文件名，默认自动生成
            
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.html"
        
        output_path = self.output_dir / filename
        
        html_content = self._generate_html(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported: {output_path}")
        
        return str(output_path)
    
    def _generate_html(self, report: Dict) -> str:
        """
        生成HTML内容
        
        Args:
            report: 报告数据
            
        Returns:
            HTML字符串
        """
        stats = report["statistics"]
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>审计{report["report_type"]} - {report["period_name"]}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
        }}
        .meta-info {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .stat-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .stat-card.info {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-success {{
            background-color: #4CAF50;
            color: white;
        }}
        .badge-danger {{
            background-color: #f44336;
            color: white;
        }}
        .badge-warning {{
            background-color: #ff9800;
            color: white;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 审计{report["report_type"]}</h1>
        
        <div class="meta-info">
            <p><strong>报告周期：</strong>{report["period_name"]}</p>
            <p><strong>生成时间：</strong>{datetime.fromisoformat(report["generated_at"]).strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <h2>📈 总览</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">审计事件总数</div>
                <div class="stat-value">{stats["total_events"]}</div>
            </div>
            <div class="stat-card success">
                <div class="stat-label">成功率</div>
                <div class="stat-value">{stats["success_rate"] * 100:.1f}%</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">失败事件</div>
                <div class="stat-value">{stats["failed_count"]}</div>
            </div>
            <div class="stat-card info">
                <div class="stat-label">PII检测率</div>
                <div class="stat-value">{stats["pii_detection_rate"] * 100:.1f}%</div>
            </div>
        </div>
        
        <h2>👥 活跃用户 Top 10</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>用户ID</th>
                    <th>事件数量</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for idx, (user_id, count) in enumerate(stats["top_users"][:10], 1):
            html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{user_id}</td>
                    <td>{count}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>🔍 热门操作 Top 10</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>操作类型</th>
                    <th>执行次数</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for idx, (action, count) in enumerate(stats["top_actions"][:10], 1):
            html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{action}</td>
                    <td>{count}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>📋 事件类型分布</h2>
        <table>
            <thead>
                <tr>
                    <th>事件类型</th>
                    <th>数量</th>
                    <th>占比</th>
                </tr>
            </thead>
            <tbody>
"""
        
        total = stats["total_events"]
        for event_type, count in stats["event_types"].items():
            percentage = (count / total * 100) if total > 0 else 0
            html += f"""
                <tr>
                    <td>{event_type}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        # 失败事件
        if stats["failed_events"]:
            html += """
        <h2>⚠️ 失败事件详情（最多20条）</h2>
        <table>
            <thead>
                <tr>
                    <th>时间</th>
                    <th>用户</th>
                    <th>操作</th>
                    <th>资源</th>
                    <th>结果</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for event in stats["failed_events"]:
                timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%m-%d %H:%M")
                result_badge = "badge-danger" if event["result"] == "error" else "badge-warning"
                html += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{event["user_id"]}</td>
                    <td>{event["action"]}</td>
                    <td>{event["resource"]}</td>
                    <td><span class="badge {result_badge}">{event["result"]}</span></td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
"""
        
        html += f"""
        <div class="footer">
            <p>本报告由麒麟量化审计系统自动生成</p>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def export_json(
        self,
        report: Dict,
        filename: Optional[str] = None
    ) -> str:
        """
        导出JSON报告
        
        Args:
            report: 报告数据
            filename: 文件名
            
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_report_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON report exported: {output_path}")
        
        return str(output_path)


# 便捷函数
def generate_daily_report_html(
    audit_logger: AuditLogger,
    output_dir: str = "reports/audit"
) -> str:
    """生成并导出昨天的日报HTML"""
    generator = AuditReportGenerator(audit_logger, output_dir)
    report = generator.generate_daily_report()
    return generator.export_html(report)


def generate_weekly_report_html(
    audit_logger: AuditLogger,
    output_dir: str = "reports/audit"
) -> str:
    """生成并导出上周的周报HTML"""
    generator = AuditReportGenerator(audit_logger, output_dir)
    report = generator.generate_weekly_report()
    return generator.export_html(report)


def generate_monthly_report_html(
    audit_logger: AuditLogger,
    output_dir: str = "reports/audit"
) -> str:
    """生成并导出上月的月报HTML"""
    generator = AuditReportGenerator(audit_logger, output_dir)
    report = generator.generate_monthly_report()
    return generator.export_html(report)
