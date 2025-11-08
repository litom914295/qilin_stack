#!/usr/bin/env python
"""
深度因果分析器 - AI超级训练核心
重点分析首板次日大涨/涨停/连板的成功案例
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import math


def _to_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float(default)
        return float(str(x).replace('%','').strip())
    except Exception:
        return float(default)


def _time_to_minutes(val) -> Optional[int]:
    """将多种时间表示统一为分钟。
    支持: 'HH:MM', 'HH:MM:SS', 930/1030(整型HHMM), 0-1440(分钟), float NaN -> None
    """
    if val is None:
        return None
    # already minutes
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        v = int(val)
        if 0 <= v <= 24*60:
            return v
        if 0 <= v <= 2359:
            h, m = v // 100, v % 100
            if 0 <= h < 24 and 0 <= m < 60:
                return h * 60 + m
        return None
    if isinstance(val, str):
        s = val.strip()
        for fmt in ("%H:%M", "%H:%M:%S", "%H%M"):
            try:
                t = datetime.strptime(s, fmt)
                return t.hour * 60 + t.minute
            except Exception:
                pass
        # try numeric string
        try:
            num = int(s)
            return _time_to_minutes(num)
        except Exception:
            return None
    return None


def _format_time_display(val) -> str:
    m = _time_to_minutes(val)
    if m is None:
        return "N/A"
    h, mi = divmod(int(m), 60)
    return f"{h:02d}:{mi:02d}"


class DeepCausalityAnalyzer:
    """深度因果分析器 - 专注于成功案例的归因学习"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.causal_graph = {}  # 因果图谱
        self.success_patterns = {}  # 成功模式库
        self.failure_patterns = {}  # 失败模式库
        
        # 成功标准（核心！）
        self.SUCCESS_CRITERIA = {
            'excellent': {  # 优秀（涨停）
                'min_return': 0.095,
                'priority': 'highest',
                'weight': 3.0
            },
            'great': {  # 很好（大涨）
                'min_return': 0.05,
                'priority': 'high',
                'weight': 2.0
            },
            'good': {  # 较好（上涨）
                'min_return': 0.02,
                'priority': 'medium',
                'weight': 1.0
            },
            'mediocre': {  # 一般
                'min_return': 0.0,
                'priority': 'low',
                'weight': 0.3
            }
        }
    
    def analyze_success_case(self, stock_data: Dict, result_data: Dict) -> Dict:
        """
        分析成功案例
        
        重点关注：
        1. 首板次日继续涨停（最重要！）
        2. 首板次日大涨（>5%）
        3. 成功连板（2连板+）
        """
        
        # 1. 判断成功级别
        success_level = self._classify_success_level(result_data['return_1d'])
        
        if success_level == 'mediocre':
            # 不是真正的成功案例，跳过详细分析
            return {'success': False, 'level': 'mediocre'}
        
        # 2. 提取关键特征
        key_features = self._extract_key_features(stock_data)
        
        # 3. 构建LLM分析提示词
        analysis_prompt = self._build_success_analysis_prompt(
            stock_data, 
            result_data, 
            key_features,
            success_level
        )
        
        # 4. LLM深度分析（如果有LLM）
        if self.llm:
            llm_analysis = self._call_llm_analysis(analysis_prompt)
            causal_chain = self._extract_causal_chain(llm_analysis)
        else:
            # 基于规则的分析
            causal_chain = self._rule_based_analysis(stock_data, result_data, key_features)
        
        # 5. 识别成功模式
        pattern = self._identify_success_pattern(stock_data, causal_chain, success_level)
        
        # 6. 更新因果图谱
        self._update_causal_graph(causal_chain, success_level)
        
        # 7. 保存到成功模式库
        self._update_success_patterns(pattern, success_level)
        
        return {
            'success': True,
            'level': success_level,
            'causal_chain': causal_chain,
            'pattern': pattern,
            'key_factors': self._rank_factors_by_importance(causal_chain),
            'weight': self.SUCCESS_CRITERIA[success_level]['weight']
        }
    
    def _classify_success_level(self, next_day_return: float) -> str:
        """分类成功级别"""
        
        if next_day_return >= 0.095:  # 涨停
            return 'excellent'
        elif next_day_return >= 0.05:  # 大涨
            return 'great'
        elif next_day_return >= 0.02:  # 上涨
            return 'good'
        else:
            return 'mediocre'
    
    def _extract_key_features(self, stock_data: Dict) -> Dict:
        """提取关键特征，做类型与缺失健壮化"""
        return {
            # 涨停特征
            'limitup_time': stock_data.get('limitup_time', None),
            'seal_strength': _to_float(stock_data.get('seal_strength', 0)),
            'consecutive_days': int(_to_float(stock_data.get('consecutive_days', 1))),
            
            # 资金特征
            'main_inflow': _to_float(stock_data.get('main_inflow', 0)),
            'turnover_rate': _to_float(stock_data.get('turnover_rate', 0)),
            'volume_ratio': _to_float(stock_data.get('volume_ratio', 0)),
            
            # 板块特征
            'sector_limitup_count': int(_to_float(stock_data.get('sector_limitup_count', 0))),
            'sector_strength': _to_float(stock_data.get('sector_strength', 0)),
            'is_sector_leader': bool(stock_data.get('is_sector_leader', False)),
            
            # 题材特征
            'theme_hotness': _to_float(stock_data.get('theme_hotness', 0)),
            'theme_consecutive_days': _to_float(stock_data.get('theme_consecutive_days', 0)),
            'is_first_limitup_in_theme': bool(stock_data.get('is_first_limitup_in_theme', False)),
            
            # 市场环境
            'market_sentiment': stock_data.get('market_sentiment', 'neutral'),
            'total_limitup': int(_to_float(stock_data.get('total_limitup', 0))),
            'break_rate': _to_float(stock_data.get('break_rate', 0)),
            'max_consecutive_boards': int(_to_float(stock_data.get('max_consecutive_boards', 0))),
        }
    
    def _build_success_analysis_prompt(
        self, 
        stock_data: Dict, 
        result_data: Dict, 
        key_features: Dict,
        success_level: str
    ) -> str:
        """构建成功案例分析提示词"""
        
        success_desc = {
            'excellent': '**次日涨停**（最优秀的表现！）',
            'great': '**次日大涨**（>5%涨幅）',
            'good': '**次日上涨**（2-5%涨幅）'
        }
        
        # 时间展示与早盘标记
        limit_m = _time_to_minutes(key_features.get('limitup_time'))
        limit_disp = _format_time_display(key_features.get('limitup_time'))
        early_tag = '(早盘封板=强势)' if (isinstance(limit_m, int) and limit_m < 10*60 + 30) else ''
        seal = _to_float(key_features.get('seal_strength', 0))
        
        return f"""
你是顶级涨停板归因分析专家。请深度分析以下首板成功案例。

# 案例表现
{success_desc[success_level]}
- 次日收益率: {result_data['return_1d']:.2%}
- 3日收益率: {result_data.get('return_3d', 0):.2%}
- 5日最高涨幅: {result_data.get('max_return_5d', 0):.2%}

# 股票基本信息
- 代码: {stock_data['code']}
- 名称: {stock_data.get('name', '')}
- 日期: {stock_data['date']}
- 板块: {stock_data.get('sector', '')}
- 题材: {stock_data.get('theme', '')}

# 涨停当日关键特征
## 涨停质量
- 涨停时间: {limit_disp} {early_tag}
- 封板强度: {seal:.1f}% {'✅ 强封板' if seal > 80 else '⚠️ 弱封板'}
- 连板天数: {key_features['consecutive_days']} (首板)

## 资金情况
- 主力净流入: {key_features['main_inflow']:.0f}万 {'✅ 大资金' if key_features['main_inflow'] > 5000 else ''}
- 换手率: {key_features['turnover_rate']:.1f}%
- 量比: {key_features['volume_ratio']:.2f}

## 板块效应
- 板块涨停数: {key_features['sector_limitup_count']} {'✅ 板块效应' if key_features['sector_limitup_count'] > 3 else ''}
- 板块地位: {'✅ 龙头' if key_features['is_sector_leader'] else '跟风'}

## 题材热度
- 题材热度: {key_features['theme_hotness']} {'✅ 热门题材' if key_features['theme_hotness'] > 5 else ''}
- 题材龙头: {'✅ 首板' if key_features['is_first_limitup_in_theme'] else '跟风'}

## 市场环境
- 市场情绪: {key_features['market_sentiment']}
- 全市场涨停数: {key_features['total_limitup']}
- 炸板率: {key_features['break_rate']:.1f}%
- 连板高度: {key_features['max_consecutive_boards']}板

# 分析要求

请按以下框架深度分析**为什么这只首板次日能{success_desc[success_level].replace('**', '')}**：

1. **核心驱动因素** (最重要！)
   - 是什么根本原因让它成功？
   - 题材驱动？资金推动？技术突破？消息刺激？板块效应？

2. **关键成功要素**
   - 哪3个特征最关键？
   - 如果缺少这些特征，还会成功吗？

3. **时机选择**
   - 为什么是这一天涨停？
   - 市场环境如何配合？

4. **持续性来源**
   - 为什么次日能延续强势？
   - 资金/题材/基本面的哪个因素支撑了持续性？

5. **可复制模式**
   - 这个案例属于什么类型的成功模式？
   - 相似特征组合的历史成功率如何？

请以JSON格式输出：
{{
    "root_cause": "根本驱动因素",
    "key_success_factors": ["因素1", "因素2", "因素3"],
    "timing_reason": "时机选择原因",
    "sustainability_source": "持续性来源",
    "pattern_type": "成功模式类型",
    "replicability_score": 0-10,
    "key_insight": "核心洞察（一句话总结）"
}}
"""
    
    def _rule_based_analysis(self, stock_data: Dict, result_data: Dict, key_features: Dict) -> Dict:
        """基于规则的分析（无LLM时使用）"""
        
        causal_chain = {
            'root_cause': '',
            'key_factors': [],
            'pattern_type': ''
        }
        
        # 判断驱动类型
        if key_features['theme_hotness'] > 5 and key_features['is_first_limitup_in_theme']:
            causal_chain['root_cause'] = '题材首板龙头'
            causal_chain['pattern_type'] = 'theme_leader'
        elif key_features['sector_limitup_count'] > 5 and key_features['is_sector_leader']:
            causal_chain['root_cause'] = '板块共振龙头'
            causal_chain['pattern_type'] = 'sector_leader'
        elif key_features['main_inflow'] > 10000:
            causal_chain['root_cause'] = '大资金推动'
            causal_chain['pattern_type'] = 'capital_driven'
        else:
            causal_chain['root_cause'] = '技术突破'
            causal_chain['pattern_type'] = 'technical_breakout'
        
        # 关键因素
        factors = []
        if key_features['seal_strength'] > 80:
            factors.append('强封板')
        lm = _time_to_minutes(key_features.get('limitup_time'))
        if isinstance(lm, int) and lm < 10*60 + 30:
            factors.append('早盘封板')
        if key_features['main_inflow'] > 5000:
            factors.append('主力资金流入')
        if key_features['sector_limitup_count'] > 3:
            factors.append('板块效应')
        if key_features['theme_hotness'] > 5:
            factors.append('题材热度')
        
        causal_chain['key_factors'] = factors[:3]  # 取前3个
        
        return causal_chain
    
    def _extract_causal_chain(self, llm_analysis: str) -> Dict:
        """从LLM分析中提取因果链"""
        try:
            # 尝试解析JSON
            return json.loads(llm_analysis)
        except:
            # 解析失败，返回空结构
            return {
                'root_cause': 'unknown',
                'key_success_factors': [],
                'pattern_type': 'unknown'
            }
    
    def _identify_success_pattern(self, stock_data: Dict, causal_chain: Dict, success_level: str) -> Dict:
        """识别成功模式"""
        
        pattern = {
            'pattern_type': causal_chain.get('pattern_type', 'unknown'),
            'success_level': success_level,
            'key_features': causal_chain.get('key_success_factors', []),
            'root_cause': causal_chain.get('root_cause', ''),
            'sample_date': stock_data.get('date', ''),
            'sample_code': stock_data.get('code', ''),
            'replicability': causal_chain.get('replicability_score', 5)
        }
        
        return pattern
    
    def _rank_factors_by_importance(self, causal_chain: Dict) -> List[str]:
        """按重要性排序因素"""
        return causal_chain.get('key_success_factors', [])
    
    def _update_causal_graph(self, causal_chain: Dict, success_level: str):
        """更新因果图谱"""
        
        pattern_type = causal_chain.get('pattern_type', 'unknown')
        
        if pattern_type not in self.causal_graph:
            self.causal_graph[pattern_type] = {
                'count': 0,
                'success_by_level': {
                    'excellent': 0,
                    'great': 0,
                    'good': 0
                },
                'key_factors_frequency': {}
            }
        
        self.causal_graph[pattern_type]['count'] += 1
        self.causal_graph[pattern_type]['success_by_level'][success_level] += 1
        
        # 统计关键因素频率
        for factor in causal_chain.get('key_success_factors', []):
            if factor not in self.causal_graph[pattern_type]['key_factors_frequency']:
                self.causal_graph[pattern_type]['key_factors_frequency'][factor] = 0
            self.causal_graph[pattern_type]['key_factors_frequency'][factor] += 1
    
    def _update_success_patterns(self, pattern: Dict, success_level: str):
        """更新成功模式库"""
        
        pattern_type = pattern['pattern_type']
        
        if pattern_type not in self.success_patterns:
            self.success_patterns[pattern_type] = {
                'total_count': 0,
                'success_by_level': {
                    'excellent': 0,
                    'great': 0,
                    'good': 0
                },
                'success_rate': 0.0,
                'avg_return': 0.0,
                'samples': []
            }
        
        self.success_patterns[pattern_type]['total_count'] += 1
        self.success_patterns[pattern_type]['success_by_level'][success_level] += 1
        self.success_patterns[pattern_type]['samples'].append(pattern)
        
        # 计算成功率（优秀+很好）
        excellent = self.success_patterns[pattern_type]['success_by_level']['excellent']
        great = self.success_patterns[pattern_type]['success_by_level']['great']
        total = self.success_patterns[pattern_type]['total_count']
        
        self.success_patterns[pattern_type]['success_rate'] = (excellent + great) / total
    
    def get_success_patterns_summary(self) -> pd.DataFrame:
        """获取成功模式摘要"""
        
        summary = []
        
        for pattern_type, stats in self.success_patterns.items():
            summary.append({
                'pattern_type': pattern_type,
                'total_count': stats['total_count'],
                'excellent_count': stats['success_by_level']['excellent'],
                'great_count': stats['success_by_level']['great'],
                'good_count': stats['success_by_level']['good'],
                'success_rate': stats['success_rate'],
                'avg_return': stats.get('avg_return', 0)
            })
        
        return pd.DataFrame(summary).sort_values('success_rate', ascending=False)
    
    def get_causal_graph_summary(self) -> pd.DataFrame:
        """获取因果图谱摘要"""
        
        summary = []
        
        for pattern_type, stats in self.causal_graph.items():
            summary.append({
                'pattern_type': pattern_type,
                'count': stats['count'],
                'excellent': stats['success_by_level']['excellent'],
                'great': stats['success_by_level']['great'],
                'good': stats['success_by_level']['good'],
                'top_factors': ', '.join(
                    sorted(
                        stats['key_factors_frequency'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                )
            })
        
        return pd.DataFrame(summary).sort_values('count', ascending=False)
    
    def save(self, save_dir: Path):
        """保存分析结果"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存因果图谱
        with open(save_dir / 'causal_graph.json', 'w', encoding='utf-8') as f:
            json.dump(self.causal_graph, f, ensure_ascii=False, indent=2)
        
        # 保存成功模式
        with open(save_dir / 'success_patterns.json', 'w', encoding='utf-8') as f:
            # 简化samples（太大）
            simplified = {}
            for k, v in self.success_patterns.items():
                simplified[k] = {**v}
                simplified[k]['samples'] = simplified[k]['samples'][:10]  # 只保留前10个
            json.dump(simplified, f, ensure_ascii=False, indent=2)
    
    def load(self, save_dir: Path):
        """加载分析结果"""
        
        save_dir = Path(save_dir)
        
        if (save_dir / 'causal_graph.json').exists():
            with open(save_dir / 'causal_graph.json', 'r', encoding='utf-8') as f:
                self.causal_graph = json.load(f)
        
        if (save_dir / 'success_patterns.json').exists():
            with open(save_dir / 'success_patterns.json', 'r', encoding='utf-8') as f:
                self.success_patterns = json.load(f)
