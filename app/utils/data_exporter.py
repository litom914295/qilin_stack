"""
数据导出系统
支持多种格式的分析结果导出
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """数据导出器"""
    
    SUPPORTED_FORMATS = ['excel', 'csv', 'json', 'markdown', 'html']
    
    def __init__(self, output_dir: str = "./exports"):
        """
        初始化数据导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_analysis_result(self,
                               data: Dict[str, Any],
                               format: str = 'excel',
                               filename: Optional[str] = None) -> str:
        """
        导出分析结果
        
        Args:
            data: 分析结果数据
            format: 导出格式 (excel, csv, json, markdown, html)
            filename: 文件名（可选）
            
        Returns:
            导出文件路径
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {format}。支持的格式: {self.SUPPORTED_FORMATS}")
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_result_{timestamp}"
        
        # 根据格式导出
        if format == 'excel':
            return self._export_to_excel(data, filename)
        elif format == 'csv':
            return self._export_to_csv(data, filename)
        elif format == 'json':
            return self._export_to_json(data, filename)
        elif format == 'markdown':
            return self._export_to_markdown(data, filename)
        elif format == 'html':
            return self._export_to_html(data, filename)
    
    def _export_to_excel(self, data: Dict[str, Any], filename: str) -> str:
        """导出为Excel"""
        try:
            filepath = self.output_dir / f"{filename}.xlsx"
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 主要信息
                if 'stock_code' in data:
                    summary_df = pd.DataFrame([{
                        '股票代码': data.get('stock_code'),
                        '分析时间': data.get('analysis_time'),
                        '市场': data.get('market'),
                    }])
                    summary_df.to_excel(writer, sheet_name='概要', index=False)
                
                # 基本面分析
                if 'fundamental_analysis' in data:
                    fund = data['fundamental_analysis']
                    fund_df = pd.DataFrame([fund.get('key_metrics', {})])
                    fund_df['评分'] = fund.get('score')
                    fund_df['建议'] = fund.get('recommendation')
                    fund_df.to_excel(writer, sheet_name='基本面', index=False)
                
                # 技术面分析
                if 'technical_analysis' in data:
                    tech = data['technical_analysis']
                    tech_df = pd.DataFrame([tech.get('indicators', {})])
                    tech_df['评分'] = tech.get('score')
                    tech_df['趋势'] = tech.get('trend')
                    tech_df.to_excel(writer, sheet_name='技术面', index=False)
                
                # 新闻情绪
                if 'news_sentiment' in data:
                    news = data['news_sentiment']
                    news_df = pd.DataFrame([{
                        '评分': news.get('score'),
                        '情绪': news.get('sentiment'),
                        '新闻数量': news.get('news_count')
                    }])
                    news_df.to_excel(writer, sheet_name='新闻情绪', index=False)
                
                # 最终决策
                if 'final_decision' in data:
                    decision = data['final_decision']
                    decision_df = pd.DataFrame([decision])
                    decision_df.to_excel(writer, sheet_name='决策', index=False)
                
                # 因子数据（如果有）
                if 'factors' in data and isinstance(data['factors'], list):
                    factors_df = pd.DataFrame(data['factors'])
                    factors_df.to_excel(writer, sheet_name='因子', index=False)
            
            logger.info(f"成功导出Excel文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出Excel失败: {e}")
            raise
    
    def _export_to_csv(self, data: Dict[str, Any], filename: str) -> str:
        """导出为CSV"""
        try:
            # 转换为DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame({'data': [str(data)]})
            
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            logger.info(f"成功导出CSV文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            raise
    
    def _export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """导出为JSON"""
        try:
            filepath = self.output_dir / f"{filename}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"成功导出JSON文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出JSON失败: {e}")
            raise
    
    def _export_to_markdown(self, data: Dict[str, Any], filename: str) -> str:
        """导出为Markdown"""
        try:
            filepath = self.output_dir / f"{filename}.md"
            
            md_content = self._generate_markdown(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"成功导出Markdown文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出Markdown失败: {e}")
            raise
    
    def _generate_markdown(self, data: Dict[str, Any]) -> str:
        """生成Markdown内容"""
        lines = []
        
        # 标题
        lines.append(f"# 股票分析报告\n")
        
        # 概要信息
        if 'stock_code' in data:
            lines.append(f"## 基本信息\n")
            lines.append(f"- **股票代码**: {data.get('stock_code')}")
            lines.append(f"- **分析时间**: {data.get('analysis_time')}")
            lines.append(f"- **市场**: {data.get('market')}\n")
        
        # 基本面分析
        if 'fundamental_analysis' in data:
            fund = data['fundamental_analysis']
            lines.append(f"## 基本面分析\n")
            lines.append(f"- **评分**: {fund.get('score')}")
            lines.append(f"- **建议**: {fund.get('recommendation')}")
            
            if 'key_metrics' in fund:
                lines.append(f"\n### 关键指标")
                for key, value in fund['key_metrics'].items():
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # 技术面分析
        if 'technical_analysis' in data:
            tech = data['technical_analysis']
            lines.append(f"## 技术面分析\n")
            lines.append(f"- **评分**: {tech.get('score')}")
            lines.append(f"- **趋势**: {tech.get('trend')}")
            
            if 'indicators' in tech:
                lines.append(f"\n### 技术指标")
                for key, value in tech['indicators'].items():
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # 新闻情绪
        if 'news_sentiment' in data:
            news = data['news_sentiment']
            lines.append(f"## 新闻情绪分析\n")
            lines.append(f"- **评分**: {news.get('score')}")
            lines.append(f"- **情绪**: {news.get('sentiment')}")
            lines.append(f"- **新闻数量**: {news.get('news_count')}\n")
        
        # 最终决策
        if 'final_decision' in data:
            decision = data['final_decision']
            lines.append(f"## 投资决策\n")
            lines.append(f"- **操作建议**: {decision.get('action')}")
            lines.append(f"- **信心度**: {decision.get('confidence', 0):.2%}")
            lines.append(f"- **目标价**: ¥{decision.get('target_price', 0):.2f}")
            lines.append(f"- **风险等级**: {decision.get('risk_level')}\n")
        
        return "\n".join(lines)
    
    def _export_to_html(self, data: Dict[str, Any], filename: str) -> str:
        """导出为HTML"""
        try:
            filepath = self.output_dir / f"{filename}.html"
            
            html_content = self._generate_html(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"成功导出HTML文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出HTML失败: {e}")
            raise
    
    def _generate_html(self, data: Dict[str, Any]) -> str:
        """生成HTML内容"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>股票分析报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .info-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin: 10px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            color: #333;
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
        }}
        .decision {{
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .decision-action {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 股票分析报告</h1>
        
        <div class="info-box">
            <div class="metric">
                <span class="metric-label">股票代码:</span>
                <span class="metric-value">{data.get('stock_code', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">分析时间:</span>
                <span class="metric-value">{data.get('analysis_time', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">市场:</span>
                <span class="metric-value">{data.get('market', 'N/A')}</span>
            </div>
        </div>
        """
        
        # 添加分析内容
        if 'final_decision' in data:
            decision = data['final_decision']
            html += f"""
        <div class="decision">
            <h2>💡 投资决策</h2>
            <div class="decision-action">{decision.get('action', 'N/A')}</div>
            <p>
                <span class="metric-label">信心度:</span> {decision.get('confidence', 0):.2%} |
                <span class="metric-label">目标价:</span> ¥{decision.get('target_price', 0):.2f} |
                <span class="metric-label">风险等级:</span> {decision.get('risk_level', 'N/A')}
            </p>
        </div>
        """
        
        html += """
    </div>
</body>
</html>
        """
        
        return html
    
    def export_batch_results(self,
                            results: List[Dict[str, Any]],
                            format: str = 'excel',
                            filename: Optional[str] = None) -> str:
        """
        导出批量分析结果
        
        Args:
            results: 批量分析结果
            format: 导出格式
            filename: 文件名
            
        Returns:
            导出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_analysis_{timestamp}"
        
        if format == 'excel':
            return self._export_batch_to_excel(results, filename)
        elif format == 'csv':
            return self._export_batch_to_csv(results, filename)
        else:
            return self._export_to_json(results, filename)
    
    def _export_batch_to_excel(self, results: List[Dict], filename: str) -> str:
        """批量导出到Excel"""
        try:
            filepath = self.output_dir / f"{filename}.xlsx"
            
            # 汇总表
            summary_data = []
            for result in results:
                summary_data.append({
                    '股票代码': result.get('stock_code'),
                    '操作建议': result.get('final_decision', {}).get('action'),
                    '信心度': result.get('final_decision', {}).get('confidence'),
                    '目标价': result.get('final_decision', {}).get('target_price'),
                    '风险等级': result.get('final_decision', {}).get('risk_level'),
                    '基本面评分': result.get('fundamental_analysis', {}).get('score'),
                    '技术面评分': result.get('technical_analysis', {}).get('score'),
                    '新闻情绪评分': result.get('news_sentiment', {}).get('score'),
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='汇总', index=False)
            
            logger.info(f"成功导出批量Excel文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出批量Excel失败: {e}")
            raise
    
    def _export_batch_to_csv(self, results: List[Dict], filename: str) -> str:
        """批量导出到CSV"""
        try:
            # 汇总表
            summary_data = []
            for result in results:
                summary_data.append({
                    '股票代码': result.get('stock_code'),
                    '操作建议': result.get('final_decision', {}).get('action'),
                    '信心度': result.get('final_decision', {}).get('confidence'),
                    '目标价': result.get('final_decision', {}).get('target_price'),
                    '风险等级': result.get('final_decision', {}).get('risk_level'),
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            filepath = self.output_dir / f"{filename}.csv"
            summary_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            logger.info(f"成功导出批量CSV文件: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"导出批量CSV失败: {e}")
            raise


# 全局导出器实例
data_exporter = DataExporter()
