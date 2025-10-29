"""
麒麟量化系统 - 板块题材管理系统
负责板块映射、热度计算、龙头识别
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional
from pathlib import Path
import logging
import json
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class SectorThemeManager:
    """板块题材管理器"""
    
    def __init__(self, theme_map_path: str = "data/theme/theme_map.csv"):
        """
        初始化板块管理器
        
        Args:
            theme_map_path: 股票-板块映射文件路径
        """
        self.theme_map_path = theme_map_path
        self.stock_to_themes: Dict[str, List[str]] = {}  # 股票 -> 板块列表
        self.theme_to_stocks: Dict[str, Set[str]] = defaultdict(set)  # 板块 -> 股票集合
        
        self._load_theme_map()
        
        logger.info(f"板块管理器初始化完成: {len(self.stock_to_themes)} 只股票, "
                   f"{len(self.theme_to_stocks)} 个板块")
    
    def _load_theme_map(self):
        """加载板块映射表"""
        path = Path(self.theme_map_path)
        
        if not path.exists():
            logger.warning(f"板块映射文件不存在: {path}, 将创建示例文件")
            self._create_sample_theme_map()
            return
        
        try:
            df = pd.read_csv(path, encoding='utf-8')
            
            if 'instrument' not in df.columns or 'theme' not in df.columns:
                logger.error("映射文件格式错误,需要包含 instrument 和 theme 列")
                return
            
            for _, row in df.iterrows():
                symbol = str(row['instrument']).strip()
                themes_str = str(row['theme']).strip()
                
                if not symbol or not themes_str or themes_str == 'nan':
                    continue
                
                # 支持多个板块用分号分隔
                themes = [t.strip() for t in themes_str.split(';') if t.strip()]
                
                if themes:
                    self.stock_to_themes[symbol] = themes
                    
                    for theme in themes:
                        self.theme_to_stocks[theme].add(symbol)
            
            logger.info(f"成功加载板块映射: {len(self.stock_to_themes)} 只股票")
            
        except Exception as e:
            logger.error(f"加载板块映射失败: {e}")
    
    def _create_sample_theme_map(self):
        """创建示例板块映射文件"""
        sample_data = [
            # 新能源板块
            {"instrument": "300750", "theme": "新能源;锂电池;汽车零部件"},
            {"instrument": "002594", "theme": "新能源;光伏;半导体"},
            {"instrument": "688599", "theme": "新能源;锂电池;储能"},
            
            # 科技板块
            {"instrument": "000063", "theme": "科技;通信;5G"},
            {"instrument": "002415", "theme": "科技;半导体;芯片"},
            {"instrument": "688981", "theme": "科技;人工智能;半导体"},
            
            # 医药板块
            {"instrument": "300142", "theme": "医药;生物医药;疫苗"},
            {"instrument": "600276", "theme": "医药;中药;医疗服务"},
            {"instrument": "688139", "theme": "医药;创新药;医疗器械"},
            
            # 消费板块
            {"instrument": "600519", "theme": "消费;白酒;食品饮料"},
            {"instrument": "000858", "theme": "消费;家电;智能家居"},
            {"instrument": "002304", "theme": "消费;教育;传媒"},
            
            # 金融板块
            {"instrument": "000001", "theme": "金融;银行;保险"},
            {"instrument": "600036", "theme": "金融;银行;资产管理"},
            {"instrument": "601318", "theme": "金融;保险;投资"},
            
            # 地产建筑
            {"instrument": "000002", "theme": "地产;房地产;物业管理"},
            {"instrument": "600048", "theme": "地产;基建;水利工程"},
            
            # 军工板块
            {"instrument": "000768", "theme": "军工;航空航天;国防"},
            {"instrument": "002013", "theme": "军工;船舶;海洋工程"},
        ]
        
        df = pd.DataFrame(sample_data)
        
        # 确保目录存在
        path = Path(self.theme_map_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path, index=False, encoding='utf-8')
        logger.info(f"已创建示例板块映射文件: {path}")
        
        # 重新加载
        self._load_theme_map()
    
    def get_themes(self, symbol: str) -> List[str]:
        """获取股票所属板块"""
        return self.stock_to_themes.get(symbol, [])
    
    def get_stocks_in_theme(self, theme: str) -> Set[str]:
        """获取板块内所有股票"""
        return self.theme_to_stocks.get(theme, set())
    
    def calculate_theme_heat(
        self, 
        limit_up_stocks: List[str],
        date: Optional[str] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        计算板块热度
        
        Args:
            limit_up_stocks: 当日涨停股票列表
            date: 日期(可选)
            
        Returns:
            板块热度字典: {theme: {count, heat, members}}
        """
        theme_counts = defaultdict(int)
        theme_members = defaultdict(list)
        
        # 统计每个板块的涨停数
        for symbol in limit_up_stocks:
            themes = self.get_themes(symbol)
            for theme in themes:
                theme_counts[theme] += 1
                theme_members[theme].append(symbol)
        
        if not theme_counts:
            return {}
        
        # 计算热度(归一化到0-1)
        max_count = max(theme_counts.values())
        
        result = {}
        for theme, count in theme_counts.items():
            heat = count / max(max_count, 1)  # 归一化
            
            result[theme] = {
                "count": count,
                "heat": min(heat, 1.0),
                "members": theme_members[theme],
                "theme_name": theme
            }
        
        # 按热度排序
        result = dict(sorted(
            result.items(), 
            key=lambda x: x[1]["count"], 
            reverse=True
        ))
        
        logger.info(f"计算板块热度完成: {len(result)} 个热门板块")
        
        return result
    
    def identify_sector_leader(
        self,
        theme: str,
        candidates: List[Dict],
        criteria: str = "quality_score"
    ) -> Optional[str]:
        """
        识别板块龙头
        
        Args:
            theme: 板块名称
            candidates: 候选股票列表(需包含symbol和评分)
            criteria: 排序标准(quality_score/auction_strength等)
            
        Returns:
            龙头股票代码
        """
        theme_stocks = self.get_stocks_in_theme(theme)
        
        # 筛选出该板块的候选股票
        theme_candidates = [
            c for c in candidates 
            if c.get("symbol") in theme_stocks
        ]
        
        if not theme_candidates:
            return None
        
        # 按指定标准排序
        theme_candidates.sort(
            key=lambda x: x.get(criteria, 0), 
            reverse=True
        )
        
        leader = theme_candidates[0]["symbol"]
        
        logger.debug(f"板块 {theme} 龙头: {leader}")
        
        return leader
    
    def get_stock_sector_info(
        self,
        symbol: str,
        limit_up_stocks: List[str]
    ) -> Dict[str, any]:
        """
        获取股票板块相关信息
        
        Args:
            symbol: 股票代码
            limit_up_stocks: 当日涨停股票列表
            
        Returns:
            板块信息字典
        """
        themes = self.get_themes(symbol)
        
        if not themes:
            return {
                "has_theme": False,
                "themes": [],
                "max_theme_heat": 0.0,
                "max_theme_count": 0,
                "is_leader": False
            }
        
        # 计算板块热度
        theme_heat_map = self.calculate_theme_heat(limit_up_stocks)
        
        # 找到该股票所在板块中热度最高的
        max_heat = 0.0
        max_count = 0
        hottest_theme = None
        
        for theme in themes:
            if theme in theme_heat_map:
                heat_info = theme_heat_map[theme]
                if heat_info["heat"] > max_heat:
                    max_heat = heat_info["heat"]
                    max_count = heat_info["count"]
                    hottest_theme = theme
        
        # 判断是否龙头
        is_leader = False
        if hottest_theme and hottest_theme in theme_heat_map:
            members = theme_heat_map[hottest_theme]["members"]
            # 简单判断: 如果是该板块涨停股中的第一个,视为龙头
            if members and members[0] == symbol:
                is_leader = True
        
        return {
            "has_theme": True,
            "themes": themes,
            "hottest_theme": hottest_theme,
            "max_theme_heat": max_heat,
            "max_theme_count": max_count,
            "is_leader": is_leader
        }
    
    def export_daily_heat_report(
        self,
        limit_up_stocks: List[str],
        output_path: str = "reports/sector_heat"
    ) -> str:
        """
        导出每日板块热度报告
        
        Args:
            limit_up_stocks: 涨停股票列表
            output_path: 输出目录
            
        Returns:
            报告文件路径
        """
        heat_map = self.calculate_theme_heat(limit_up_stocks)
        
        if not heat_map:
            logger.warning("没有板块热度数据")
            return ""
        
        # 生成报告
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_limit_up": len(limit_up_stocks),
            "hot_sectors": []
        }
        
        for theme, info in heat_map.items():
            report["hot_sectors"].append({
                "theme": theme,
                "count": info["count"],
                "heat": round(info["heat"], 3),
                "members": info["members"]
            })
        
        # 保存
        Path(output_path).mkdir(parents=True, exist_ok=True)
        filename = f"{output_path}/sector_heat_{datetime.now():%Y%m%d}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"板块热度报告已保存: {filename}")
        
        return filename
    
    def add_stock_theme(self, symbol: str, themes: List[str]):
        """动态添加股票板块映射"""
        self.stock_to_themes[symbol] = themes
        
        for theme in themes:
            self.theme_to_stocks[theme].add(symbol)
    
    def save_theme_map(self):
        """保存板块映射到文件"""
        data = []
        for symbol, themes in self.stock_to_themes.items():
            data.append({
                "instrument": symbol,
                "theme": ";".join(themes)
            })
        
        df = pd.DataFrame(data)
        path = Path(self.theme_map_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path, index=False, encoding='utf-8')
        logger.info(f"板块映射已保存: {path}")


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    manager = SectorThemeManager()
    
    # 模拟涨停股票
    limit_up_stocks = [
        "300750", "002594", "688599",  # 新能源
        "002415", "688981",             # 科技
        "600519"                        # 消费
    ]
    
    print("\n=== 板块热度计算 ===")
    heat = manager.calculate_theme_heat(limit_up_stocks)
    for theme, info in list(heat.items())[:5]:
        print(f"{theme}: {info['count']}只, 热度{info['heat']:.2f}")
    
    print("\n=== 股票板块信息 ===")
    info = manager.get_stock_sector_info("300750", limit_up_stocks)
    print(f"宁德时代: {info}")
    
    print("\n=== 导出热度报告 ===")
    report_path = manager.export_daily_heat_report(limit_up_stocks)
    print(f"报告路径: {report_path}")
