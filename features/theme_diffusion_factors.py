"""
题材扩散与龙头因子系统

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务1.8
目标：捕捉热点题材和龙头战法，评估题材扩散度和龙头强度

核心维度：
1. 题材识别：热门题材、题材涨停数、题材市值
2. 题材扩散：题材集中度、轮动速度、扩散广度
3. 龙头识别：龙头股、龙头溢价、龙头稳定性
4. 题材生命周期：新题材、成熟题材、衰退题材
5. 板块联动：行业联动度、板块共振
6. 题材联动性：题材间相关性、共振强度
7. 跨板块扩散：扩散路径、影响范围、扩散速度
8. 龙头接力：龙头切换、接力连续性、接力强度

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ThemeDiffusionFactors:
    """题材扩散与龙头因子计算器"""
    
    # 预定义常见题材关键词
    THEME_KEYWORDS = {
        'AI人工智能': ['人工智能', 'AI', 'chatgpt', '大模型', '算力', 'GPU'],
        '新能源': ['新能源', '锂电', '光伏', '风电', '储能', '氢能'],
        '半导体': ['半导体', '芯片', '集成电路', '晶圆', '封测'],
        '军工': ['军工', '航天', '航空', '船舶', '兵器'],
        '医药': ['医药', '生物', '疫苗', '创新药', 'CXO'],
        '消费': ['消费', '白酒', '食品', '零售', '餐饮'],
        '金融': ['银行', '保险', '券商', '信托'],
        '地产': ['房地产', '建筑', '装修', '家居'],
        '5G通信': ['5G', '通信', '物联网', '基站'],
        '元宇宙': ['元宇宙', 'VR', 'AR', '虚拟现实'],
        '数字经济': ['数字经济', '大数据', '云计算', '区块链'],
        '碳中和': ['碳中和', '环保', '节能', '清洁能源'],
        '国企改革': ['国企改革', '央企', '混改'],
        '一带一路': ['一带一路', '基建', '出口'],
        '乡村振兴': ['乡村振兴', '农业', '种业']
    }
    
    def __init__(self):
        """初始化题材扩散因子计算器"""
        self.theme_cache = {}
        self.leader_cache = {}
        self.theme_history = {}  # 历史题材数据
        self.leader_history = {}  # 历史龙头数据
        print("🎯 题材扩散与龙头因子计算器初始化")
    
    def calculate_all_factors(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        计算所有题材扩散和龙头因子
        
        Args:
            date: 交易日期
            market_data: 市场数据（必须包含stock_name, concept等字段）
        
        Returns:
            Dict: 包含所有题材和龙头因子的字典
        """
        print(f"\n计算 {date} 题材扩散与龙头因子...")
        
        factors = {}
        
        # 1. 题材识别与分类
        theme_analysis = self.analyze_themes(date, market_data)
        factors.update(theme_analysis)
        
        # 2. 题材扩散度分析
        diffusion_analysis = self.calculate_theme_diffusion(date, theme_analysis)
        factors.update(diffusion_analysis)
        
        # 3. 龙头识别与分析
        leader_analysis = self.identify_and_analyze_leaders(date, market_data, theme_analysis)
        factors.update(leader_analysis)
        
        # 4. 题材生命周期
        lifecycle_analysis = self.analyze_theme_lifecycle(date, theme_analysis)
        factors.update(lifecycle_analysis)
        
        # 5. 板块联动分析
        sector_analysis = self.analyze_sector_linkage(date, market_data)
        factors.update(sector_analysis)
        
        # 6. 题材联动性分析
        theme_linkage = self.analyze_theme_linkage(date, theme_analysis)
        factors.update(theme_linkage)
        
        # 7. 跨板块扩散分析
        cross_sector_diffusion = self.analyze_cross_sector_diffusion(date, market_data, theme_analysis)
        factors.update(cross_sector_diffusion)
        
        # 8. 龙头接力关系分析
        leader_relay = self.analyze_leader_relay(date, leader_analysis)
        factors.update(leader_relay)
        
        # 缓存历史数据
        self.theme_history[date] = theme_analysis
        self.leader_history[date] = leader_analysis
        
        print(f"✅ 共计算 {len(factors)} 个题材与龙头因子")
        
        return factors
    
    def analyze_themes(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        题材识别与分类
        
        识别当日热门题材，统计每个题材的涨停数、市值等
        """
        print("  分析热门题材...")
        
        factors = {}
        
        try:
            # 获取涨停股票数据
            limitup_stocks = self._get_limitup_stocks(date, market_data)
            
            if limitup_stocks is not None and not limitup_stocks.empty:
                # 提取题材信息
                theme_stats = self._extract_theme_statistics(limitup_stocks)
                
                # 1. 热门题材Top 5
                top_themes = sorted(theme_stats.items(), key=lambda x: x[1]['limitup_count'], reverse=True)[:5]
                
                for i, (theme_name, stats) in enumerate(top_themes, 1):
                    factors[f'top_{i}_theme_name'] = theme_name
                    factors[f'top_{i}_theme_limitup_count'] = stats['limitup_count']
                    factors[f'top_{i}_theme_avg_seal_strength'] = stats['avg_seal_strength']
                    factors[f'top_{i}_theme_total_market_cap'] = stats['total_market_cap']
                
                # 填充剩余位置
                for i in range(len(top_themes) + 1, 6):
                    factors[f'top_{i}_theme_name'] = '无'
                    factors[f'top_{i}_theme_limitup_count'] = 0
                    factors[f'top_{i}_theme_avg_seal_strength'] = 0
                    factors[f'top_{i}_theme_total_market_cap'] = 0
                
                # 2. 题材总数
                factors['total_active_themes'] = len(theme_stats)
                
                # 3. 最强题材（涨停数最多）
                if top_themes:
                    strongest_theme = top_themes[0]
                    factors['strongest_theme_name'] = strongest_theme[0]
                    factors['strongest_theme_dominance'] = strongest_theme[1]['limitup_count'] / len(limitup_stocks)
                else:
                    factors['strongest_theme_name'] = '无'
                    factors['strongest_theme_dominance'] = 0
                
                # 4. 题材涨停股占比
                total_limitup = len(limitup_stocks)
                themed_limitup = sum(stats['limitup_count'] for stats in theme_stats.values())
                factors['themed_limitup_ratio'] = themed_limitup / total_limitup if total_limitup > 0 else 0
                
                # 缓存题材数据供后续使用
                self.theme_cache[date] = theme_stats
                
            else:
                # 无数据时填充默认值
                for i in range(1, 6):
                    factors[f'top_{i}_theme_name'] = '无'
                    factors[f'top_{i}_theme_limitup_count'] = 0
                    factors[f'top_{i}_theme_avg_seal_strength'] = 0
                    factors[f'top_{i}_theme_total_market_cap'] = 0
                
                factors.update({
                    'total_active_themes': 0,
                    'strongest_theme_name': '无',
                    'strongest_theme_dominance': 0,
                    'themed_limitup_ratio': 0
                })
        
        except Exception as e:
            print(f"    ⚠️ 题材分析失败: {e}")
            for i in range(1, 6):
                factors[f'top_{i}_theme_name'] = '无'
                factors[f'top_{i}_theme_limitup_count'] = 0
                factors[f'top_{i}_theme_avg_seal_strength'] = 0
                factors[f'top_{i}_theme_total_market_cap'] = 0
            
            factors.update({
                'total_active_themes': 0,
                'strongest_theme_name': '无',
                'strongest_theme_dominance': 0,
                'themed_limitup_ratio': 0
            })
        
        return factors
    
    def calculate_theme_diffusion(self, date: str, theme_analysis: Dict) -> Dict:
        """
        计算题材扩散度
        
        评估题材的集中度、扩散广度、轮动速度
        """
        print("  计算题材扩散度...")
        
        factors = {}
        
        try:
            theme_stats = self.theme_cache.get(date, {})
            
            if theme_stats:
                # 1. 题材集中度（HHI指数）
                # HHI = Σ(市场份额^2)，越高越集中
                total_limitup = sum(stats['limitup_count'] for stats in theme_stats.values())
                
                if total_limitup > 0:
                    hhi = sum((stats['limitup_count'] / total_limitup) ** 2 for stats in theme_stats.values())
                    factors['theme_concentration_hhi'] = hhi
                    
                    # HHI解释：
                    # >0.25: 高度集中（一个题材独大）
                    # 0.15-0.25: 中度集中
                    # <0.15: 分散（多题材共存）
                    if hhi > 0.25:
                        factors['theme_concentration_level'] = '高度集中'
                    elif hhi > 0.15:
                        factors['theme_concentration_level'] = '中度集中'
                    else:
                        factors['theme_concentration_level'] = '分散'
                else:
                    factors['theme_concentration_hhi'] = 0
                    factors['theme_concentration_level'] = '无'
                
                # 2. 题材扩散广度（有涨停股的题材数/总题材数）
                factors['theme_diffusion_breadth'] = len(theme_stats)
                
                # 3. 题材平均涨停数
                if theme_stats:
                    factors['avg_limitup_per_theme'] = total_limitup / len(theme_stats)
                else:
                    factors['avg_limitup_per_theme'] = 0
                
                # 4. 龙头题材集中度（Top3题材占比）
                sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['limitup_count'], reverse=True)
                top3_limitup = sum(stats['limitup_count'] for _, stats in sorted_themes[:3])
                factors['top3_theme_concentration'] = top3_limitup / total_limitup if total_limitup > 0 else 0
                
                # 5. 题材轮动速度（与前一天对比）
                rotation_speed = self._calculate_theme_rotation_speed(date, theme_stats)
                factors['theme_rotation_speed'] = rotation_speed
                
            else:
                factors.update({
                    'theme_concentration_hhi': 0,
                    'theme_concentration_level': '无',
                    'theme_diffusion_breadth': 0,
                    'avg_limitup_per_theme': 0,
                    'top3_theme_concentration': 0,
                    'theme_rotation_speed': 0
                })
        
        except Exception as e:
            print(f"    ⚠️ 题材扩散度计算失败: {e}")
            factors.update({
                'theme_concentration_hhi': 0,
                'theme_concentration_level': '无',
                'theme_diffusion_breadth': 0,
                'avg_limitup_per_theme': 0,
                'top3_theme_concentration': 0,
                'theme_rotation_speed': 0
            })
        
        return factors
    
    def identify_and_analyze_leaders(self, date: str, market_data: pd.DataFrame = None, 
                                    theme_analysis: Dict = None) -> Dict:
        """
        识别并分析龙头股
        
        龙头识别标准：
        1. 连板高度高
        2. 封单强度大
        3. 市值适中（100-500亿）
        4. 成交活跃
        5. 题材纯正
        """
        print("  识别和分析龙头股...")
        
        factors = {}
        
        try:
            limitup_stocks = self._get_limitup_stocks(date, market_data)
            
            if limitup_stocks is not None and not limitup_stocks.empty:
                # 龙头候选：连板数>=2的股票
                if 'consecutive_days' in limitup_stocks.columns:
                    leader_candidates = limitup_stocks[limitup_stocks['consecutive_days'] >= 2].copy()
                else:
                    leader_candidates = limitup_stocks.copy()
                
                if not leader_candidates.empty:
                    # 计算龙头评分
                    leader_scores = self._calculate_leader_scores(leader_candidates)
                    leader_candidates['leader_score'] = leader_scores
                    
                    # 按评分排序，取Top 5
                    top_leaders = leader_candidates.nlargest(5, 'leader_score')
                    
                    # 1. Top 5龙头信息
                    for i, (idx, row) in enumerate(top_leaders.iterrows(), 1):
                        factors[f'leader_{i}_name'] = row.get('name', f'股票{i}')
                        factors[f'leader_{i}_consecutive_days'] = int(row.get('consecutive_days', 1))
                        factors[f'leader_{i}_seal_strength'] = float(row.get('seal_strength', 0))
                        factors[f'leader_{i}_score'] = float(row.get('leader_score', 0))
                    
                    # 填充剩余位置
                    for i in range(len(top_leaders) + 1, 6):
                        factors[f'leader_{i}_name'] = '无'
                        factors[f'leader_{i}_consecutive_days'] = 0
                        factors[f'leader_{i}_seal_strength'] = 0
                        factors[f'leader_{i}_score'] = 0
                    
                    # 2. 龙头数量
                    factors['total_leader_count'] = len(leader_candidates)
                    
                    # 3. 最强龙头高度
                    max_consecutive = leader_candidates['consecutive_days'].max() if 'consecutive_days' in leader_candidates.columns else 1
                    factors['max_leader_height'] = int(max_consecutive)
                    
                    # 4. 龙头平均封单强度
                    if 'seal_strength' in leader_candidates.columns:
                        factors['leader_avg_seal_strength'] = float(leader_candidates['seal_strength'].mean())
                    else:
                        factors['leader_avg_seal_strength'] = 0
                    
                    # 5. 龙头溢价（龙头封单强度 / 市场平均封单强度）
                    if 'seal_strength' in limitup_stocks.columns:
                        market_avg_seal = limitup_stocks['seal_strength'].mean()
                        leader_avg_seal = leader_candidates['seal_strength'].mean()
                        factors['leader_premium'] = leader_avg_seal / market_avg_seal if market_avg_seal > 0 else 1.0
                    else:
                        factors['leader_premium'] = 1.0
                    
                    # 6. 龙头稳定性（连板>=3的龙头数量）
                    if 'consecutive_days' in leader_candidates.columns:
                        stable_leader_count = (leader_candidates['consecutive_days'] >= 3).sum()
                        factors['stable_leader_count'] = int(stable_leader_count)
                        factors['stable_leader_ratio'] = stable_leader_count / len(leader_candidates)
                    else:
                        factors['stable_leader_count'] = 0
                        factors['stable_leader_ratio'] = 0
                    
                    # 缓存龙头数据
                    self.leader_cache[date] = top_leaders
                    
                else:
                    # 无龙头候选
                    self._fill_no_leader_factors(factors)
            else:
                self._fill_no_leader_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 龙头分析失败: {e}")
            self._fill_no_leader_factors(factors)
        
        return factors
    
    def analyze_theme_lifecycle(self, date: str, theme_analysis: Dict) -> Dict:
        """
        分析题材生命周期
        
        判断题材处于：新生期、成长期、成熟期、衰退期
        """
        print("  分析题材生命周期...")
        
        factors = {}
        
        try:
            theme_stats = self.theme_cache.get(date, {})
            
            if theme_stats:
                # 与历史对比，判断题材生命周期
                lifecycle_analysis = {}
                
                for theme_name, stats in theme_stats.items():
                    # 简化实现：根据涨停数和封单强度判断
                    limitup_count = stats['limitup_count']
                    avg_seal = stats['avg_seal_strength']
                    
                    if limitup_count >= 10 and avg_seal > 5:
                        lifecycle = '成熟期'  # 高涨停数+高封单强度
                    elif limitup_count >= 5 and avg_seal > 3:
                        lifecycle = '成长期'  # 中等涨停数+中等封单强度
                    elif limitup_count >= 3:
                        lifecycle = '新生期'  # 刚起步
                    else:
                        lifecycle = '衰退期'  # 低涨停数
                    
                    lifecycle_analysis[theme_name] = lifecycle
                
                # 统计各阶段题材数
                lifecycle_counter = Counter(lifecycle_analysis.values())
                factors['emerging_theme_count'] = lifecycle_counter.get('新生期', 0)
                factors['growing_theme_count'] = lifecycle_counter.get('成长期', 0)
                factors['mature_theme_count'] = lifecycle_counter.get('成熟期', 0)
                factors['declining_theme_count'] = lifecycle_counter.get('衰退期', 0)
                
                # 主流题材生命周期
                top_theme_name = theme_analysis.get('strongest_theme_name', '无')
                factors['main_theme_lifecycle'] = lifecycle_analysis.get(top_theme_name, '无')
                
            else:
                factors.update({
                    'emerging_theme_count': 0,
                    'growing_theme_count': 0,
                    'mature_theme_count': 0,
                    'declining_theme_count': 0,
                    'main_theme_lifecycle': '无'
                })
        
        except Exception as e:
            print(f"    ⚠️ 生命周期分析失败: {e}")
            factors.update({
                'emerging_theme_count': 0,
                'growing_theme_count': 0,
                'mature_theme_count': 0,
                'declining_theme_count': 0,
                'main_theme_lifecycle': '无'
            })
        
        return factors
    
    def analyze_sector_linkage(self, date: str, market_data: pd.DataFrame = None) -> Dict:
        """
        分析板块联动
        
        评估不同板块/行业之间的联动效应
        """
        print("  分析板块联动...")
        
        factors = {}
        
        try:
            if market_data is not None and 'sector' in market_data.columns:
                # 统计各板块涨停数
                limitup_by_sector = market_data[market_data.get('is_limit_up', 0) == 1].groupby('sector').size()
                
                if not limitup_by_sector.empty:
                    # 1. 涨停板块数
                    factors['limitup_sector_count'] = len(limitup_by_sector)
                    
                    # 2. 最强板块
                    top_sector = limitup_by_sector.idxmax()
                    factors['strongest_sector'] = top_sector
                    factors['strongest_sector_limitup_count'] = int(limitup_by_sector.max())
                    
                    # 3. 板块集中度
                    total_sector_limitup = limitup_by_sector.sum()
                    sector_hhi = sum((count / total_sector_limitup) ** 2 for count in limitup_by_sector.values)
                    factors['sector_concentration_hhi'] = sector_hhi
                    
                    # 4. 板块轮动（Top3板块占比）
                    top3_sector_limitup = limitup_by_sector.nlargest(3).sum()
                    factors['top3_sector_ratio'] = top3_sector_limitup / total_sector_limitup if total_sector_limitup > 0 else 0
                    
                    # 5. 板块平均涨停数
                    factors['avg_limitup_per_sector'] = float(limitup_by_sector.mean())
                    
                else:
                    self._fill_no_sector_factors(factors)
            else:
                self._fill_no_sector_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 板块联动分析失败: {e}")
            self._fill_no_sector_factors(factors)
        
        return factors
    
    def analyze_theme_linkage(self, date: str, theme_analysis: Dict) -> Dict:
        """
        分析题材间的联动性
        
        评估不同题材之间的相关性和共振效应
        """
        print("  分析题材联动性...")
        
        factors = {}
        
        try:
            theme_stats = self.theme_cache.get(date, {})
            
            if len(theme_stats) >= 2:
                # 1. 计算题材股票重叠度
                # 检查不同题材之间有多少股票是共同的（一股多题材）
                theme_stocks = {theme: set(stats['stocks']) for theme, stats in theme_stats.items()}
                
                # Top 5 题材间的重叠度矩阵
                top_themes = sorted(theme_stats.items(), key=lambda x: x[1]['limitup_count'], reverse=True)[:5]
                overlap_scores = []
                
                for i in range(len(top_themes)):
                    for j in range(i + 1, len(top_themes)):
                        theme1, theme2 = top_themes[i][0], top_themes[j][0]
                        stocks1 = theme_stocks.get(theme1, set())
                        stocks2 = theme_stocks.get(theme2, set())
                        
                        if stocks1 and stocks2:
                            # Jaccard相似度
                            intersection = len(stocks1 & stocks2)
                            union = len(stocks1 | stocks2)
                            overlap = intersection / union if union > 0 else 0
                            overlap_scores.append(overlap)
                
                # 2. 平均题材联动强度
                if overlap_scores:
                    factors['theme_linkage_strength'] = np.mean(overlap_scores)
                    factors['theme_max_linkage'] = np.max(overlap_scores)
                    
                    # 联动级别分类
                    avg_linkage = factors['theme_linkage_strength']
                    if avg_linkage > 0.4:
                        factors['theme_linkage_level'] = '强联动'
                    elif avg_linkage > 0.2:
                        factors['theme_linkage_level'] = '中等联动'
                    else:
                        factors['theme_linkage_level'] = '弱联动'
                else:
                    factors['theme_linkage_strength'] = 0
                    factors['theme_max_linkage'] = 0
                    factors['theme_linkage_level'] = '无联动'
                
                # 3. 共振题材对数量（重叠度>0.3的题材对）
                resonance_pairs = sum(1 for score in overlap_scores if score > 0.3)
                factors['theme_resonance_pairs'] = resonance_pairs
                
                # 4. 题材共振强度（有多少题材在共同发力）
                # 如果多个题材涨停数都很高，说明共振强
                strong_themes = [t for t, s in theme_stats.items() if s['limitup_count'] >= 5]
                factors['strong_theme_count'] = len(strong_themes)
                
                # 5. 题材共振比例
                factors['theme_resonance_ratio'] = len(strong_themes) / len(theme_stats) if theme_stats else 0
                
            else:
                # 题材数量不足，无法计算联动
                factors.update({
                    'theme_linkage_strength': 0,
                    'theme_max_linkage': 0,
                    'theme_linkage_level': '无联动',
                    'theme_resonance_pairs': 0,
                    'strong_theme_count': 0,
                    'theme_resonance_ratio': 0
                })
        
        except Exception as e:
            print(f"    ⚠️ 题材联动性分析失败: {e}")
            factors.update({
                'theme_linkage_strength': 0,
                'theme_max_linkage': 0,
                'theme_linkage_level': '无联动',
                'theme_resonance_pairs': 0,
                'strong_theme_count': 0,
                'theme_resonance_ratio': 0
            })
        
        return factors
    
    def analyze_cross_sector_diffusion(self, date: str, market_data: pd.DataFrame = None,
                                       theme_analysis: Dict = None) -> Dict:
        """
        分析跨板块扩散路径
        
        评估题材从某个板块向其他板块扩散的情况
        """
        print("  分析跨板块扩散...")
        
        factors = {}
        
        try:
            limitup_stocks = self._get_limitup_stocks(date, market_data)
            theme_stats = self.theme_cache.get(date, {})
            
            if limitup_stocks is not None and not limitup_stocks.empty and 'sector' in limitup_stocks.columns:
                # 1. 统计每个题材涉及的板块数
                theme_sector_spread = {}
                
                for theme, stats in theme_stats.items():
                    theme_stock_names = set(stats['stocks'])
                    # 找到这些股票所属的板块
                    theme_limitup = limitup_stocks[limitup_stocks.get('name', limitup_stocks.index).isin(theme_stock_names)]
                    
                    if not theme_limitup.empty and 'sector' in theme_limitup.columns:
                        sectors = theme_limitup['sector'].nunique()
                        theme_sector_spread[theme] = sectors
                
                # 2. 最广扩散题材（涉及板块最多）
                if theme_sector_spread:
                    most_spread_theme = max(theme_sector_spread, key=theme_sector_spread.get)
                    factors['most_spread_theme'] = most_spread_theme
                    factors['most_spread_sector_count'] = theme_sector_spread[most_spread_theme]
                else:
                    factors['most_spread_theme'] = '无'
                    factors['most_spread_sector_count'] = 0
                
                # 3. 平均板块扩散度
                if theme_sector_spread:
                    factors['avg_sector_spread'] = np.mean(list(theme_sector_spread.values()))
                else:
                    factors['avg_sector_spread'] = 0
                
                # 4. 跨板块扩散强度（涉及3+板块的题材占比）
                if theme_sector_spread:
                    cross_sector_themes = sum(1 for count in theme_sector_spread.values() if count >= 3)
                    factors['cross_sector_theme_count'] = cross_sector_themes
                    factors['cross_sector_theme_ratio'] = cross_sector_themes / len(theme_sector_spread)
                else:
                    factors['cross_sector_theme_count'] = 0
                    factors['cross_sector_theme_ratio'] = 0
                
                # 5. 扩散路径分析（主导板块 -> 跟随板块）
                # 找到涨停数最多的板块，判断其他板块是否跟随
                sector_limitup_count = limitup_stocks.groupby('sector').size()
                
                if not sector_limitup_count.empty:
                    dominant_sector = sector_limitup_count.idxmax()
                    factors['dominant_diffusion_sector'] = dominant_sector
                    
                    # 计算跟随板块数（涨停数>=3的其他板块）
                    following_sectors = (sector_limitup_count >= 3).sum() - 1  # 减去主导板块
                    factors['following_sector_count'] = max(0, following_sectors)
                    
                    # 扩散效率（跟随板块占比）
                    total_sectors = len(sector_limitup_count)
                    factors['diffusion_efficiency'] = following_sectors / (total_sectors - 1) if total_sectors > 1 else 0
                else:
                    factors['dominant_diffusion_sector'] = '无'
                    factors['following_sector_count'] = 0
                    factors['diffusion_efficiency'] = 0
                
                # 6. 扩散速度（新增板块数 vs 昨日）
                # 简化实现：返回当前涉及板块总数作为代理指标
                factors['current_diffusion_breadth'] = limitup_stocks['sector'].nunique() if 'sector' in limitup_stocks.columns else 0
                
            else:
                self._fill_no_diffusion_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 跨板块扩散分析失败: {e}")
            self._fill_no_diffusion_factors(factors)
        
        return factors
    
    def analyze_leader_relay(self, date: str, leader_analysis: Dict) -> Dict:
        """
        分析龙头接力关系
        
        评估龙头股的切换、接力连续性和强度
        """
        print("  分析龙头接力关系...")
        
        factors = {}
        
        try:
            current_leaders = self.leader_cache.get(date, pd.DataFrame())
            
            if not current_leaders.empty:
                # 1. 龙头接力连续性
                # 检查是否有龙头连续多日保持龙头地位
                if 'consecutive_days' in current_leaders.columns:
                    max_consecutive = current_leaders['consecutive_days'].max()
                    
                    # 超高连板（>=5）
                    super_leader_count = (current_leaders['consecutive_days'] >= 5).sum()
                    factors['super_leader_count'] = int(super_leader_count)
                    
                    # 龙头持续性指标
                    factors['leader_continuity_score'] = float(current_leaders['consecutive_days'].mean())
                    
                    # 连续性级别
                    if max_consecutive >= 7:
                        factors['leader_continuity_level'] = '超强持续'
                    elif max_consecutive >= 5:
                        factors['leader_continuity_level'] = '强持续'
                    elif max_consecutive >= 3:
                        factors['leader_continuity_level'] = '中等持续'
                    else:
                        factors['leader_continuity_level'] = '弱持续'
                else:
                    factors['super_leader_count'] = 0
                    factors['leader_continuity_score'] = 1.0
                    factors['leader_continuity_level'] = '未知'
                
                # 2. 龙头接力强度
                # 如果有多个龙头同时在高位（连板>=3），说明接力强
                if 'consecutive_days' in current_leaders.columns:
                    high_leaders = (current_leaders['consecutive_days'] >= 3).sum()
                    total_leaders = len(current_leaders)
                    
                    factors['high_level_leader_count'] = int(high_leaders)
                    factors['leader_relay_strength'] = high_leaders / total_leaders if total_leaders > 0 else 0
                    
                    # 接力强度级别
                    relay_strength = factors['leader_relay_strength']
                    if relay_strength > 0.6:
                        factors['leader_relay_level'] = '强接力'
                    elif relay_strength > 0.3:
                        factors['leader_relay_level'] = '中等接力'
                    else:
                        factors['leader_relay_level'] = '弱接力'
                else:
                    factors['high_level_leader_count'] = 0
                    factors['leader_relay_strength'] = 0
                    factors['leader_relay_level'] = '无接力'
                
                # 3. 龙头梯队完整性
                # 理想情况：既有高位龙头（5+板），也有中位龙头（3-4板），还有低位龙头（2板）
                if 'consecutive_days' in current_leaders.columns:
                    high_tier = (current_leaders['consecutive_days'] >= 5).sum()
                    mid_tier = ((current_leaders['consecutive_days'] >= 3) & (current_leaders['consecutive_days'] < 5)).sum()
                    low_tier = (current_leaders['consecutive_days'] == 2).sum()
                    
                    factors['leader_high_tier_count'] = int(high_tier)
                    factors['leader_mid_tier_count'] = int(mid_tier)
                    factors['leader_low_tier_count'] = int(low_tier)
                    
                    # 梯队完整性：三个梯队都有则完整
                    tier_completeness = (high_tier > 0) + (mid_tier > 0) + (low_tier > 0)
                    factors['leader_tier_completeness'] = tier_completeness / 3.0
                    
                    if tier_completeness == 3:
                        factors['leader_tier_structure'] = '完整梯队'
                    elif tier_completeness == 2:
                        factors['leader_tier_structure'] = '部分梯队'
                    else:
                        factors['leader_tier_structure'] = '单一梯队'
                else:
                    factors['leader_high_tier_count'] = 0
                    factors['leader_mid_tier_count'] = 0
                    factors['leader_low_tier_count'] = 0
                    factors['leader_tier_completeness'] = 0
                    factors['leader_tier_structure'] = '无梯队'
                
                # 4. 龙头切换分析
                # 对比前一日龙头，看是否有新龙头出现
                prev_date = self._get_previous_trade_date(date)
                prev_leaders = self.leader_cache.get(prev_date, pd.DataFrame())
                
                if not prev_leaders.empty and 'name' in current_leaders.columns and 'name' in prev_leaders.columns:
                    current_names = set(current_leaders['name'].values)
                    prev_names = set(prev_leaders['name'].values)
                    
                    # 新增龙头
                    new_leaders = current_names - prev_names
                    factors['new_leader_count'] = len(new_leaders)
                    
                    # 持续龙头（两天都在）
                    continuing_leaders = current_names & prev_names
                    factors['continuing_leader_count'] = len(continuing_leaders)
                    
                    # 龙头稳定性（持续龙头占比）
                    factors['leader_stability'] = len(continuing_leaders) / len(current_names) if current_names else 0
                    
                    # 龙头切换率
                    factors['leader_turnover_rate'] = len(new_leaders) / len(current_names) if current_names else 0
                    
                    # 切换模式
                    if factors['leader_turnover_rate'] > 0.7:
                        factors['leader_switch_mode'] = '快速切换'
                    elif factors['leader_turnover_rate'] > 0.4:
                        factors['leader_switch_mode'] = '正常轮动'
                    else:
                        factors['leader_switch_mode'] = '稳定持续'
                else:
                    factors['new_leader_count'] = 0
                    factors['continuing_leader_count'] = 0
                    factors['leader_stability'] = 0
                    factors['leader_turnover_rate'] = 0
                    factors['leader_switch_mode'] = '未知'
                
            else:
                self._fill_no_relay_factors(factors)
        
        except Exception as e:
            print(f"    ⚠️ 龙头接力分析失败: {e}")
            self._fill_no_relay_factors(factors)
        
        return factors
    
    # ==================== 辅助方法 ====================
    
    def _get_limitup_stocks(self, date: str, market_data: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """获取涨停股票数据"""
        if market_data is not None:
            # 筛选涨停股票
            if 'is_limit_up' in market_data.columns:
                return market_data[market_data['is_limit_up'] == 1].copy()
            else:
                return market_data.copy()
        
        # 尝试从外部获取
        try:
            import akshare as ak
            date_str = date.replace('-', '')
            df = ak.stock_zt_pool_em(date=date_str)
            
            if not df.empty:
                df['is_limit_up'] = 1
                return df
        except:
            pass
        
        return None
    
    def _extract_theme_statistics(self, limitup_stocks: pd.DataFrame) -> Dict:
        """从涨停股票中提取题材统计"""
        theme_stats = defaultdict(lambda: {
            'limitup_count': 0,
            'total_market_cap': 0,
            'seal_strengths': [],
            'stocks': []
        })
        
        for idx, row in limitup_stocks.iterrows():
            # 尝试从概念/名称中识别题材
            themes = self._identify_themes_from_stock(row)
            
            for theme in themes:
                theme_stats[theme]['limitup_count'] += 1
                theme_stats[theme]['total_market_cap'] += row.get('market_cap', 0)
                theme_stats[theme]['seal_strengths'].append(row.get('seal_strength', 0))
                theme_stats[theme]['stocks'].append(row.get('name', ''))
        
        # 计算平均封单强度
        for theme, stats in theme_stats.items():
            if stats['seal_strengths']:
                stats['avg_seal_strength'] = np.mean(stats['seal_strengths'])
            else:
                stats['avg_seal_strength'] = 0
            # 清理临时数据
            del stats['seal_strengths']
        
        return dict(theme_stats)
    
    def _identify_themes_from_stock(self, stock_row: pd.Series) -> List[str]:
        """从股票信息中识别题材"""
        themes = []
        
        # 从名称识别
        stock_name = str(stock_row.get('name', '')).lower()
        
        # 从概念识别
        concept = str(stock_row.get('concept', '')).lower()
        
        # 合并文本
        text = stock_name + ' ' + concept
        
        # 匹配预定义题材
        for theme_name, keywords in self.THEME_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    themes.append(theme_name)
                    break
        
        # 如果没有匹配到，返回"其他"
        if not themes:
            themes.append('其他')
        
        return themes
    
    def _calculate_theme_rotation_speed(self, date: str, current_theme_stats: Dict) -> float:
        """计算题材轮动速度"""
        # 简化实现：返回默认值
        # 实际应该对比前一天的题材排名变化
        return 0.5  # 0-1之间，越高轮动越快
    
    def _calculate_leader_scores(self, candidates: pd.DataFrame) -> pd.Series:
        """
        计算龙头评分
        
        评分维度：
        1. 连板高度（40%）
        2. 封单强度（30%）
        3. 市值适中性（15%）
        4. 换手率（15%）
        """
        scores = pd.Series(index=candidates.index, dtype=float)
        
        # 1. 连板高度得分（归一化）
        if 'consecutive_days' in candidates.columns:
            consecutive_norm = (candidates['consecutive_days'] - candidates['consecutive_days'].min()) / \
                              (candidates['consecutive_days'].max() - candidates['consecutive_days'].min() + 1e-6)
            score_consecutive = consecutive_norm * 40
        else:
            score_consecutive = 20  # 默认中等分
        
        # 2. 封单强度得分
        if 'seal_strength' in candidates.columns:
            seal_norm = candidates['seal_strength'] / (candidates['seal_strength'].max() + 1e-6)
            score_seal = seal_norm * 30
        else:
            score_seal = 15
        
        # 3. 市值适中性得分（100-500亿最佳）
        if 'market_cap' in candidates.columns:
            market_cap_billion = candidates['market_cap'] / 1e8
            # 使用高斯函数，峰值在300亿
            score_market_cap = 15 * np.exp(-((market_cap_billion - 300) / 200) ** 2)
        else:
            score_market_cap = 7.5
        
        # 4. 换手率得分（5-15%最佳）
        if 'turnover_rate' in candidates.columns:
            turnover_opt = 10  # 最佳换手率
            score_turnover = 15 * np.exp(-((candidates['turnover_rate'] - turnover_opt) / 10) ** 2)
        else:
            score_turnover = 7.5
        
        scores = score_consecutive + score_seal + score_market_cap + score_turnover
        
        return scores
    
    def _fill_no_leader_factors(self, factors: Dict):
        """填充无龙头时的默认值"""
        for i in range(1, 6):
            factors[f'leader_{i}_name'] = '无'
            factors[f'leader_{i}_consecutive_days'] = 0
            factors[f'leader_{i}_seal_strength'] = 0
            factors[f'leader_{i}_score'] = 0
        
        factors.update({
            'total_leader_count': 0,
            'max_leader_height': 0,
            'leader_avg_seal_strength': 0,
            'leader_premium': 1.0,
            'stable_leader_count': 0,
            'stable_leader_ratio': 0
        })
    
    def _fill_no_sector_factors(self, factors: Dict):
        """填充无板块数据时的默认值"""
        factors.update({
            'limitup_sector_count': 0,
            'strongest_sector': '无',
            'strongest_sector_limitup_count': 0,
            'sector_concentration_hhi': 0,
            'top3_sector_ratio': 0,
            'avg_limitup_per_sector': 0
        })
    
    def _fill_no_diffusion_factors(self, factors: Dict):
        """填充无扩散数据时的默认值"""
        factors.update({
            'most_spread_theme': '无',
            'most_spread_sector_count': 0,
            'avg_sector_spread': 0,
            'cross_sector_theme_count': 0,
            'cross_sector_theme_ratio': 0,
            'dominant_diffusion_sector': '无',
            'following_sector_count': 0,
            'diffusion_efficiency': 0,
            'current_diffusion_breadth': 0
        })
    
    def _fill_no_relay_factors(self, factors: Dict):
        """填充无接力数据时的默认值"""
        factors.update({
            'super_leader_count': 0,
            'leader_continuity_score': 0,
            'leader_continuity_level': '无',
            'high_level_leader_count': 0,
            'leader_relay_strength': 0,
            'leader_relay_level': '无接力',
            'leader_high_tier_count': 0,
            'leader_mid_tier_count': 0,
            'leader_low_tier_count': 0,
            'leader_tier_completeness': 0,
            'leader_tier_structure': '无梯队',
            'new_leader_count': 0,
            'continuing_leader_count': 0,
            'leader_stability': 0,
            'leader_turnover_rate': 0,
            'leader_switch_mode': '未知'
        })
    
    def _get_previous_trade_date(self, date: str) -> str:
        """获取前一个交易日"""
        # 简化实现：直接减1天
        # 实际应该查询交易日历
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            prev_date = date_obj - timedelta(days=1)
            return prev_date.strftime('%Y-%m-%d')
        except:
            return date


def main():
    """主函数 - 示例用法"""
    calculator = ThemeDiffusionFactors()
    
    # 计算今日题材扩散
    today = datetime.now().strftime('%Y-%m-%d')
    factors = calculator.calculate_all_factors(today)
    
    print("\n" + "="*70)
    print("🎯 题材扩散与龙头因子计算结果")
    print("="*70)
    
    # 热门题材
    print("\n【热门题材 Top 5】")
    for i in range(1, 6):
        theme_name = factors.get(f'top_{i}_theme_name', '无')
        limitup_count = factors.get(f'top_{i}_theme_limitup_count', 0)
        if theme_name != '无':
            print(f"  {i}. {theme_name}: {limitup_count}只涨停")
    
    # 题材扩散
    print("\n【题材扩散度】")
    print(f"  题材集中度(HHI): {factors.get('theme_concentration_hhi', 0):.4f}")
    print(f"  集中度级别: {factors.get('theme_concentration_level', '无')}")
    print(f"  活跃题材数: {factors.get('total_active_themes', 0)}")
    print(f"  题材轮动速度: {factors.get('theme_rotation_speed', 0):.2f}")
    
    # 龙头股
    print("\n【龙头股 Top 5】")
    for i in range(1, 6):
        leader_name = factors.get(f'leader_{i}_name', '无')
        consecutive = factors.get(f'leader_{i}_consecutive_days', 0)
        score = factors.get(f'leader_{i}_score', 0)
        if leader_name != '无':
            print(f"  {i}. {leader_name}: {consecutive}连板, 评分{score:.1f}")
    
    print(f"\n  龙头总数: {factors.get('total_leader_count', 0)}")
    print(f"  最高连板: {factors.get('max_leader_height', 0)}")
    print(f"  龙头溢价: {factors.get('leader_premium', 1.0):.2f}倍")
    
    # 生命周期
    print("\n【题材生命周期】")
    print(f"  新生期题材: {factors.get('emerging_theme_count', 0)}")
    print(f"  成长期题材: {factors.get('growing_theme_count', 0)}")
    print(f"  成熟期题材: {factors.get('mature_theme_count', 0)}")
    print(f"  衰退期题材: {factors.get('declining_theme_count', 0)}")
    
    # 板块联动
    print("\n【板块联动】")
    print(f"  最强板块: {factors.get('strongest_sector', '无')}")
    print(f"  板块集中度: {factors.get('sector_concentration_hhi', 0):.4f}")
    
    # 题材联动
    print("\n【题材联动性】")
    print(f"  联动强度: {factors.get('theme_linkage_strength', 0):.4f}")
    print(f"  联动级别: {factors.get('theme_linkage_level', '无')}")
    print(f"  共振题材对: {factors.get('theme_resonance_pairs', 0)}")
    print(f"  强势题材数: {factors.get('strong_theme_count', 0)}")
    
    # 跨板块扩散
    print("\n【跨板块扩散】")
    print(f"  最广扩散题材: {factors.get('most_spread_theme', '无')}")
    print(f"  涉及板块数: {factors.get('most_spread_sector_count', 0)}")
    print(f"  平均板块扩散: {factors.get('avg_sector_spread', 0):.2f}")
    print(f"  跨板块题材数: {factors.get('cross_sector_theme_count', 0)}")
    print(f"  主导扩散板块: {factors.get('dominant_diffusion_sector', '无')}")
    print(f"  跟随板块数: {factors.get('following_sector_count', 0)}")
    
    # 龙头接力
    print("\n【龙头接力关系】")
    print(f"  超强龙头数(5+板): {factors.get('super_leader_count', 0)}")
    print(f"  龙头持续性: {factors.get('leader_continuity_score', 0):.2f} ({factors.get('leader_continuity_level', '无')})")
    print(f"  高位龙头数(3+板): {factors.get('high_level_leader_count', 0)}")
    print(f"  接力强度: {factors.get('leader_relay_strength', 0):.2%} ({factors.get('leader_relay_level', '无')})")
    print(f"  梯队结构: {factors.get('leader_tier_structure', '无')} (高{factors.get('leader_high_tier_count', 0)}/中{factors.get('leader_mid_tier_count', 0)}/低{factors.get('leader_low_tier_count', 0)})")
    print(f"  新增龙头: {factors.get('new_leader_count', 0)}")
    print(f"  持续龙头: {factors.get('continuing_leader_count', 0)}")
    print(f"  龙头稳定性: {factors.get('leader_stability', 0):.2%}")
    print(f"  切换模式: {factors.get('leader_switch_mode', '未知')}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
