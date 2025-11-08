"""
题材消息面热度因子系统

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务扩展
目标：量化评估题材相关的新闻、公告、社交媒体热度

核心维度：
1. 新闻热度：题材相关新闻数量、媒体关注度、报道趋势
2. 公告热度：相关公司公告数量、利好/利空分布
3. 社交媒体热度：微博/雪球/东方财富吧讨论热度
4. 热度趋势：热度变化速度、持续时间、爆发强度
5. 情绪分析：消息面情绪倾向（正面/负面/中性）
6. 市场验证度：消息热度与股价走势的一致性

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
import re
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ThemeNewsSentimentFactors:
    """题材消息面热度因子计算器"""
    
    # 预定义题材关键词（与theme_diffusion_factors.py保持一致）
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
    
    # 情绪词典
    POSITIVE_WORDS = [
        '上涨', '暴涨', '涨停', '大涨', '飙升', '突破', '利好', '增长', 
        '创新高', '翻倍', '强势', '火爆', '热门', '龙头', '机会',
        '看好', '乐观', '积极', '赚钱', '盈利', '受益', '推动'
    ]
    
    NEGATIVE_WORDS = [
        '下跌', '暴跌', '跌停', '大跌', '重挫', '破位', '利空', '亏损',
        '创新低', '腰斩', '弱势', '冷门', '风险', '看空', '悲观',
        '警惕', '担忧', '恐慌', '抛售', '出逃', '清仓'
    ]
    
    def __init__(self):
        """初始化题材消息面热度因子计算器"""
        self.news_cache = {}  # 新闻缓存
        self.sentiment_history = {}  # 情绪历史
        print("📰 题材消息面热度因子计算器初始化")
    
    def calculate_all_factors(self, date: str, 
                             theme_name: str = None,
                             stock_code: str = None) -> Dict:
        """
        计算所有消息面热度因子
        
        Args:
            date: 日期
            theme_name: 题材名称（可选）
            stock_code: 个股代码（可选）
        
        Returns:
            Dict: 包含所有消息面因子的字典
        """
        print(f"\n计算 {date} 消息面热度因子...")
        
        factors = {}
        
        # 1. 新闻热度分析
        news_heat = self.analyze_news_heat(date, theme_name)
        factors.update(news_heat)
        
        # 2. 公告热度分析
        announcement_heat = self.analyze_announcement_heat(date, theme_name, stock_code)
        factors.update(announcement_heat)
        
        # 3. 社交媒体热度分析
        social_heat = self.analyze_social_media_heat(date, theme_name, stock_code)
        factors.update(social_heat)
        
        # 4. 热度趋势分析
        heat_trend = self.analyze_heat_trend(date, theme_name)
        factors.update(heat_trend)
        
        # 5. 情绪分析
        sentiment = self.analyze_sentiment(date, theme_name)
        factors.update(sentiment)
        
        # 6. 市场验证度
        validation = self.analyze_market_validation(date, theme_name, stock_code)
        factors.update(validation)
        
        # 缓存数据
        cache_key = f"{date}_{theme_name or 'all'}_{stock_code or ''}"
        self.sentiment_history[cache_key] = factors
        
        print(f"✅ 共计算 {len(factors)} 个消息面热度因子")
        
        return factors
    
    def analyze_news_heat(self, date: str, theme_name: str = None) -> Dict:
        """
        新闻热度分析
        
        统计题材相关新闻数量、媒体关注度等
        """
        print("  分析新闻热度...")
        
        factors = {}
        
        try:
            # 获取新闻数据
            news_data = self._get_news_data(date, theme_name)
            
            if news_data:
                # 1. 新闻数量
                factors['news_count'] = news_data.get('count', 0)
                
                # 2. 新闻增长率（vs 昨日）
                yesterday_news = self._get_news_data(
                    (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'),
                    theme_name
                )
                
                if yesterday_news and yesterday_news.get('count', 0) > 0:
                    factors['news_growth_rate'] = (news_data['count'] - yesterday_news['count']) / yesterday_news['count']
                else:
                    factors['news_growth_rate'] = 0
                
                # 3. 媒体权威度（加权平均）
                # 主流媒体（新华社、人民网等）权重高
                factors['media_authority_score'] = news_data.get('authority_score', 0)
                
                # 4. 新闻覆盖度（不同媒体数量）
                factors['media_coverage_count'] = news_data.get('media_count', 0)
                
                # 5. 头条新闻数（标题含题材关键词的核心新闻）
                factors['headline_news_count'] = news_data.get('headline_count', 0)
                
                # 6. 新闻热度评分（0-100）
                heat_score = self._calculate_news_heat_score(news_data)
                factors['news_heat_score'] = heat_score
                
                # 热度分级
                if heat_score >= 80:
                    factors['news_heat_level'] = '极热'
                elif heat_score >= 60:
                    factors['news_heat_level'] = '很热'
                elif heat_score >= 40:
                    factors['news_heat_level'] = '一般'
                elif heat_score >= 20:
                    factors['news_heat_level'] = '冷淡'
                else:
                    factors['news_heat_level'] = '极冷'
            
            else:
                self._fill_news_defaults(factors)
        
        except Exception as e:
            print(f"    ⚠️ 新闻热度分析失败: {e}")
            self._fill_news_defaults(factors)
        
        return factors
    
    def analyze_announcement_heat(self, date: str, theme_name: str = None, stock_code: str = None) -> Dict:
        """
        公告热度分析
        
        分析相关公司公告的数量和性质
        """
        print("  分析公告热度...")
        
        factors = {}
        
        try:
            # 获取公告数据
            announcement_data = self._get_announcement_data(date, theme_name, stock_code)
            
            if announcement_data:
                # 1. 公告总数
                factors['announcement_count'] = announcement_data.get('count', 0)
                
                # 2. 利好公告数
                factors['positive_announcement_count'] = announcement_data.get('positive_count', 0)
                
                # 3. 利空公告数
                factors['negative_announcement_count'] = announcement_data.get('negative_count', 0)
                
                # 4. 中性公告数
                factors['neutral_announcement_count'] = announcement_data.get('neutral_count', 0)
                
                # 5. 利好公告占比
                if factors['announcement_count'] > 0:
                    factors['positive_announcement_ratio'] = factors['positive_announcement_count'] / factors['announcement_count']
                else:
                    factors['positive_announcement_ratio'] = 0
                
                # 6. 重大公告数（业绩预告、重组、收购等）
                factors['major_announcement_count'] = announcement_data.get('major_count', 0)
                
                # 7. 公告热度评分
                ann_score = self._calculate_announcement_score(announcement_data)
                factors['announcement_heat_score'] = ann_score
                
                # 公告情绪倾向
                if factors['positive_announcement_count'] > factors['negative_announcement_count'] * 1.5:
                    factors['announcement_sentiment'] = '偏利好'
                elif factors['negative_announcement_count'] > factors['positive_announcement_count'] * 1.5:
                    factors['announcement_sentiment'] = '偏利空'
                else:
                    factors['announcement_sentiment'] = '中性'
            
            else:
                self._fill_announcement_defaults(factors)
        
        except Exception as e:
            print(f"    ⚠️ 公告热度分析失败: {e}")
            self._fill_announcement_defaults(factors)
        
        return factors
    
    def analyze_social_media_heat(self, date: str, theme_name: str = None, stock_code: str = None) -> Dict:
        """
        社交媒体热度分析
        
        分析微博、雪球、东方财富吧等平台的讨论热度
        """
        print("  分析社交媒体热度...")
        
        factors = {}
        
        try:
            # 获取社交媒体数据
            social_data = self._get_social_media_data(date, theme_name, stock_code)
            
            if social_data:
                # 1. 微博讨论数
                factors['weibo_discussion_count'] = social_data.get('weibo_count', 0)
                
                # 2. 雪球讨论数
                factors['xueqiu_discussion_count'] = social_data.get('xueqiu_count', 0)
                
                # 3. 东方财富吧讨论数
                factors['eastmoney_discussion_count'] = social_data.get('eastmoney_count', 0)
                
                # 4. 总讨论数
                factors['total_discussion_count'] = (
                    factors['weibo_discussion_count'] + 
                    factors['xueqiu_discussion_count'] + 
                    factors['eastmoney_discussion_count']
                )
                
                # 5. 讨论增长率（vs 昨日）
                yesterday_social = self._get_social_media_data(
                    (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'),
                    theme_name, stock_code
                )
                
                if yesterday_social:
                    yesterday_total = (
                        yesterday_social.get('weibo_count', 0) + 
                        yesterday_social.get('xueqiu_count', 0) + 
                        yesterday_social.get('eastmoney_count', 0)
                    )
                    
                    if yesterday_total > 0:
                        factors['social_discussion_growth_rate'] = (factors['total_discussion_count'] - yesterday_total) / yesterday_total
                    else:
                        factors['social_discussion_growth_rate'] = 0
                else:
                    factors['social_discussion_growth_rate'] = 0
                
                # 6. 热帖数（高互动帖子）
                factors['hot_post_count'] = social_data.get('hot_post_count', 0)
                
                # 7. 互动强度（点赞、评论、转发总数）
                factors['interaction_intensity'] = social_data.get('interaction_count', 0)
                
                # 8. 社交媒体热度评分
                social_score = self._calculate_social_heat_score(social_data)
                factors['social_heat_score'] = social_score
                
                # 热度分级
                if social_score >= 80:
                    factors['social_heat_level'] = '爆火'
                elif social_score >= 60:
                    factors['social_heat_level'] = '火热'
                elif social_score >= 40:
                    factors['social_heat_level'] = '温热'
                elif social_score >= 20:
                    factors['social_heat_level'] = '冷清'
                else:
                    factors['social_heat_level'] = '沉寂'
            
            else:
                self._fill_social_defaults(factors)
        
        except Exception as e:
            print(f"    ⚠️ 社交媒体热度分析失败: {e}")
            self._fill_social_defaults(factors)
        
        return factors
    
    def analyze_heat_trend(self, date: str, theme_name: str = None) -> Dict:
        """
        热度趋势分析
        
        分析热度的变化趋势、持续时间、爆发强度
        """
        print("  分析热度趋势...")
        
        factors = {}
        
        try:
            # 获取最近N天的热度数据
            recent_heat = self._get_recent_heat(date, theme_name, days=7)
            
            if recent_heat:
                # 1. 热度持续天数
                hot_days = sum(1 for h in recent_heat if h.get('heat_score', 0) >= 60)
                factors['heat_duration_days'] = hot_days
                
                # 2. 热度趋势（上升/下降/平稳）
                if len(recent_heat) >= 3:
                    recent_scores = [h.get('heat_score', 0) for h in recent_heat[:3]]
                    trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                    
                    factors['heat_trend_slope'] = float(trend_slope)
                    
                    if trend_slope > 5:
                        factors['heat_trend_direction'] = '快速上升'
                    elif trend_slope > 2:
                        factors['heat_trend_direction'] = '上升'
                    elif trend_slope < -5:
                        factors['heat_trend_direction'] = '快速下降'
                    elif trend_slope < -2:
                        factors['heat_trend_direction'] = '下降'
                    else:
                        factors['heat_trend_direction'] = '平稳'
                else:
                    factors['heat_trend_slope'] = 0
                    factors['heat_trend_direction'] = '未知'
                
                # 3. 热度爆发强度（当日热度 vs 7日平均）
                current_heat = recent_heat[0].get('heat_score', 0)
                avg_heat = np.mean([h.get('heat_score', 0) for h in recent_heat])
                
                if avg_heat > 0:
                    factors['heat_burst_intensity'] = (current_heat - avg_heat) / avg_heat
                else:
                    factors['heat_burst_intensity'] = 0
                
                # 爆发强度分级
                if factors['heat_burst_intensity'] > 1.0:
                    factors['heat_burst_level'] = '超级爆发'
                elif factors['heat_burst_intensity'] > 0.5:
                    factors['heat_burst_level'] = '强爆发'
                elif factors['heat_burst_intensity'] > 0.2:
                    factors['heat_burst_level'] = '中等爆发'
                elif factors['heat_burst_intensity'] > -0.2:
                    factors['heat_burst_level'] = '平稳'
                else:
                    factors['heat_burst_level'] = '衰退'
                
                # 4. 热度波动率
                heat_scores = [h.get('heat_score', 0) for h in recent_heat]
                factors['heat_volatility'] = float(np.std(heat_scores))
                
            else:
                factors.update({
                    'heat_duration_days': 0,
                    'heat_trend_slope': 0,
                    'heat_trend_direction': '未知',
                    'heat_burst_intensity': 0,
                    'heat_burst_level': '未知',
                    'heat_volatility': 0
                })
        
        except Exception as e:
            print(f"    ⚠️ 热度趋势分析失败: {e}")
            factors.update({
                'heat_duration_days': 0,
                'heat_trend_slope': 0,
                'heat_trend_direction': '未知',
                'heat_burst_intensity': 0,
                'heat_burst_level': '未知',
                'heat_volatility': 0
            })
        
        return factors
    
    def analyze_sentiment(self, date: str, theme_name: str = None) -> Dict:
        """
        情绪分析
        
        基于新闻、公告、社交媒体内容的情绪倾向分析
        """
        print("  分析市场情绪...")
        
        factors = {}
        
        try:
            # 获取文本数据
            text_data = self._get_text_data(date, theme_name)
            
            if text_data:
                # 1. 情绪词频统计
                positive_count = text_data.get('positive_word_count', 0)
                negative_count = text_data.get('negative_word_count', 0)
                total_words = text_data.get('total_word_count', 1)
                
                factors['positive_word_ratio'] = positive_count / total_words
                factors['negative_word_ratio'] = negative_count / total_words
                
                # 2. 情绪得分（-100到100）
                # 正面词多则为正，负面词多则为负
                sentiment_score = (positive_count - negative_count) / max(positive_count + negative_count, 1) * 100
                factors['sentiment_score'] = sentiment_score
                
                # 3. 情绪强度（正负面词总占比）
                factors['sentiment_intensity'] = (positive_count + negative_count) / total_words
                
                # 4. 情绪倾向分类
                if sentiment_score > 50:
                    factors['sentiment_tendency'] = '极度乐观'
                elif sentiment_score > 20:
                    factors['sentiment_tendency'] = '乐观'
                elif sentiment_score > -20:
                    factors['sentiment_tendency'] = '中性'
                elif sentiment_score > -50:
                    factors['sentiment_tendency'] = '悲观'
                else:
                    factors['sentiment_tendency'] = '极度悲观'
                
                # 5. 情绪一致性（不同来源情绪的一致程度）
                news_sentiment = text_data.get('news_sentiment', 0)
                social_sentiment = text_data.get('social_sentiment', 0)
                
                consistency = 1 - abs(news_sentiment - social_sentiment) / 100
                factors['sentiment_consistency'] = consistency
                
                if consistency > 0.8:
                    factors['sentiment_consistency_level'] = '高度一致'
                elif consistency > 0.6:
                    factors['sentiment_consistency_level'] = '一致'
                elif consistency > 0.4:
                    factors['sentiment_consistency_level'] = '部分一致'
                else:
                    factors['sentiment_consistency_level'] = '分歧'
            
            else:
                self._fill_sentiment_defaults(factors)
        
        except Exception as e:
            print(f"    ⚠️ 情绪分析失败: {e}")
            self._fill_sentiment_defaults(factors)
        
        return factors
    
    def analyze_market_validation(self, date: str, theme_name: str = None, stock_code: str = None) -> Dict:
        """
        市场验证度分析
        
        评估消息热度与实际股价表现的一致性
        """
        print("  分析市场验证度...")
        
        factors = {}
        
        try:
            # 获取消息热度和股价数据
            heat_score = self.sentiment_history.get(
                f"{date}_{theme_name or 'all'}_{stock_code or ''}",
                {}
            ).get('news_heat_score', 0)
            
            price_performance = self._get_price_performance(date, theme_name, stock_code)
            
            if price_performance is not None:
                # 1. 消息-涨幅一致性
                # 消息热度高且股价涨 -> 一致性高
                # 消息热度高但股价跌 -> 一致性低（可能虚假繁荣）
                
                factors['price_change_pct'] = price_performance
                
                # 归一化热度（0-1）
                normalized_heat = heat_score / 100
                # 归一化涨幅（-1到1）
                normalized_price = np.clip(price_performance / 10, -1, 1)
                
                # 一致性得分：热度和涨幅同向则高，反向则低
                validation_score = (normalized_heat * normalized_price + 1) / 2 * 100
                factors['market_validation_score'] = validation_score
                
                # 2. 验证度分级
                if validation_score > 70:
                    factors['market_validation_level'] = '高度验证'
                    factors['market_status'] = '题材有效'
                elif validation_score > 50:
                    factors['market_validation_level'] = '部分验证'
                    factors['market_status'] = '题材观察'
                elif validation_score > 30:
                    factors['market_validation_level'] = '低验证'
                    factors['market_status'] = '题材虚弱'
                else:
                    factors['market_validation_level'] = '不验证'
                    factors['market_status'] = '虚假繁荣'
                
                # 3. 超预期/不及预期
                expected_return = normalized_heat * 5  # 热度越高，预期涨幅越大
                factors['return_vs_expectation'] = price_performance - expected_return
                
                if factors['return_vs_expectation'] > 2:
                    factors['expectation_status'] = '超预期'
                elif factors['return_vs_expectation'] > -2:
                    factors['expectation_status'] = '符合预期'
                else:
                    factors['expectation_status'] = '不及预期'
            
            else:
                factors.update({
                    'price_change_pct': 0,
                    'market_validation_score': 50,
                    'market_validation_level': '未知',
                    'market_status': '未知',
                    'return_vs_expectation': 0,
                    'expectation_status': '未知'
                })
        
        except Exception as e:
            print(f"    ⚠️ 市场验证度分析失败: {e}")
            factors.update({
                'price_change_pct': 0,
                'market_validation_score': 50,
                'market_validation_level': '未知',
                'market_status': '未知',
                'return_vs_expectation': 0,
                'expectation_status': '未知'
            })
        
        return factors
    
    # ==================== 辅助方法 ====================
    
    def _get_news_data(self, date: str, theme_name: str = None) -> Optional[Dict]:
        """获取新闻数据"""
        # 简化实现：返回模拟数据
        # 实际应该调用新闻API或爬虫
        
        import random
        random.seed(hash(date + str(theme_name)))
        
        return {
            'count': random.randint(10, 100),
            'authority_score': random.uniform(50, 90),
            'media_count': random.randint(5, 30),
            'headline_count': random.randint(1, 10)
        }
    
    def _get_announcement_data(self, date: str, theme_name: str = None, stock_code: str = None) -> Optional[Dict]:
        """获取公告数据"""
        # 简化实现：返回模拟数据
        
        import random
        random.seed(hash(date + str(theme_name) + str(stock_code)))
        
        positive = random.randint(0, 5)
        negative = random.randint(0, 3)
        neutral = random.randint(0, 10)
        
        return {
            'count': positive + negative + neutral,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'major_count': random.randint(0, 3)
        }
    
    def _get_social_media_data(self, date: str, theme_name: str = None, stock_code: str = None) -> Optional[Dict]:
        """获取社交媒体数据"""
        # 简化实现：返回模拟数据
        
        import random
        random.seed(hash(date + str(theme_name) + str(stock_code)))
        
        return {
            'weibo_count': random.randint(100, 1000),
            'xueqiu_count': random.randint(50, 500),
            'eastmoney_count': random.randint(200, 2000),
            'hot_post_count': random.randint(5, 50),
            'interaction_count': random.randint(1000, 10000)
        }
    
    def _get_recent_heat(self, date: str, theme_name: str = None, days: int = 7) -> Optional[List[Dict]]:
        """获取最近N天的热度数据"""
        heat_list = []
        
        for i in range(days):
            check_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d')
            
            # 模拟热度评分
            import random
            random.seed(hash(check_date + str(theme_name)))
            
            heat_list.append({
                'date': check_date,
                'heat_score': random.uniform(30, 90)
            })
        
        return heat_list
    
    def _get_text_data(self, date: str, theme_name: str = None) -> Optional[Dict]:
        """获取文本数据（用于情绪分析）"""
        # 简化实现：返回模拟数据
        
        import random
        random.seed(hash(date + str(theme_name)))
        
        positive_count = random.randint(50, 200)
        negative_count = random.randint(20, 100)
        total_words = random.randint(1000, 5000)
        
        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'total_word_count': total_words,
            'news_sentiment': random.uniform(-50, 50),
            'social_sentiment': random.uniform(-50, 50)
        }
    
    def _get_price_performance(self, date: str, theme_name: str = None, stock_code: str = None) -> Optional[float]:
        """获取股价表现（涨跌幅%）"""
        # 简化实现：返回模拟数据
        
        import random
        random.seed(hash(date + str(theme_name) + str(stock_code)))
        
        return random.uniform(-5, 8)
    
    def _calculate_news_heat_score(self, news_data: Dict) -> float:
        """计算新闻热度评分"""
        score = 0
        
        # 新闻数量贡献（最多40分）
        score += min(news_data.get('count', 0) / 100 * 40, 40)
        
        # 媒体权威度贡献（30分）
        score += news_data.get('authority_score', 0) / 100 * 30
        
        # 媒体覆盖度贡献（20分）
        score += min(news_data.get('media_count', 0) / 30 * 20, 20)
        
        # 头条新闻贡献（10分）
        score += min(news_data.get('headline_count', 0) / 10 * 10, 10)
        
        return np.clip(score, 0, 100)
    
    def _calculate_announcement_score(self, ann_data: Dict) -> float:
        """计算公告热度评分"""
        score = 0
        
        # 公告数量贡献
        score += min(ann_data.get('count', 0) / 20 * 40, 40)
        
        # 利好公告占比贡献
        if ann_data.get('count', 0) > 0:
            score += (ann_data.get('positive_count', 0) / ann_data['count']) * 30
        
        # 重大公告贡献
        score += min(ann_data.get('major_count', 0) / 5 * 30, 30)
        
        return np.clip(score, 0, 100)
    
    def _calculate_social_heat_score(self, social_data: Dict) -> float:
        """计算社交媒体热度评分"""
        score = 0
        
        # 讨论数贡献（40分）
        total_discussion = (
            social_data.get('weibo_count', 0) + 
            social_data.get('xueqiu_count', 0) + 
            social_data.get('eastmoney_count', 0)
        )
        score += min(total_discussion / 3000 * 40, 40)
        
        # 热帖数贡献（30分）
        score += min(social_data.get('hot_post_count', 0) / 50 * 30, 30)
        
        # 互动强度贡献（30分）
        score += min(social_data.get('interaction_count', 0) / 10000 * 30, 30)
        
        return np.clip(score, 0, 100)
    
    def _fill_news_defaults(self, factors: Dict):
        """填充新闻数据默认值"""
        factors.update({
            'news_count': 0,
            'news_growth_rate': 0,
            'media_authority_score': 0,
            'media_coverage_count': 0,
            'headline_news_count': 0,
            'news_heat_score': 0,
            'news_heat_level': '极冷'
        })
    
    def _fill_announcement_defaults(self, factors: Dict):
        """填充公告数据默认值"""
        factors.update({
            'announcement_count': 0,
            'positive_announcement_count': 0,
            'negative_announcement_count': 0,
            'neutral_announcement_count': 0,
            'positive_announcement_ratio': 0,
            'major_announcement_count': 0,
            'announcement_heat_score': 0,
            'announcement_sentiment': '中性'
        })
    
    def _fill_social_defaults(self, factors: Dict):
        """填充社交媒体数据默认值"""
        factors.update({
            'weibo_discussion_count': 0,
            'xueqiu_discussion_count': 0,
            'eastmoney_discussion_count': 0,
            'total_discussion_count': 0,
            'social_discussion_growth_rate': 0,
            'hot_post_count': 0,
            'interaction_intensity': 0,
            'social_heat_score': 0,
            'social_heat_level': '沉寂'
        })
    
    def _fill_sentiment_defaults(self, factors: Dict):
        """填充情绪数据默认值"""
        factors.update({
            'positive_word_ratio': 0,
            'negative_word_ratio': 0,
            'sentiment_score': 0,
            'sentiment_intensity': 0,
            'sentiment_tendency': '中性',
            'sentiment_consistency': 0,
            'sentiment_consistency_level': '未知'
        })


def main():
    """主函数 - 示例用法"""
    calculator = ThemeNewsSentimentFactors()
    
    # 计算今日题材消息面热度
    today = datetime.now().strftime('%Y-%m-%d')
    factors = calculator.calculate_all_factors(today, theme_name='AI人工智能')
    
    print("\n" + "="*70)
    print("📰 题材消息面热度因子计算结果")
    print("="*70)
    
    # 新闻热度
    print("\n【新闻热度】")
    print(f"  新闻数量: {factors.get('news_count', 0)}")
    print(f"  新闻增长率: {factors.get('news_growth_rate', 0):.2%}")
    print(f"  媒体权威度: {factors.get('media_authority_score', 0):.1f}")
    print(f"  新闻热度: {factors.get('news_heat_score', 0):.1f} ({factors.get('news_heat_level', '未知')})")
    
    # 公告热度
    print("\n【公告热度】")
    print(f"  公告总数: {factors.get('announcement_count', 0)}")
    print(f"  利好公告: {factors.get('positive_announcement_count', 0)}")
    print(f"  利空公告: {factors.get('negative_announcement_count', 0)}")
    print(f"  公告情绪: {factors.get('announcement_sentiment', '未知')}")
    
    # 社交媒体
    print("\n【社交媒体热度】")
    print(f"  总讨论数: {factors.get('total_discussion_count', 0)}")
    print(f"  讨论增长率: {factors.get('social_discussion_growth_rate', 0):.2%}")
    print(f"  热帖数: {factors.get('hot_post_count', 0)}")
    print(f"  社交热度: {factors.get('social_heat_score', 0):.1f} ({factors.get('social_heat_level', '未知')})")
    
    # 热度趋势
    print("\n【热度趋势】")
    print(f"  热度持续: {factors.get('heat_duration_days', 0)}天")
    print(f"  趋势方向: {factors.get('heat_trend_direction', '未知')}")
    print(f"  爆发强度: {factors.get('heat_burst_level', '未知')}")
    
    # 情绪分析
    print("\n【市场情绪】")
    print(f"  情绪得分: {factors.get('sentiment_score', 0):.1f}")
    print(f"  情绪倾向: {factors.get('sentiment_tendency', '未知')}")
    print(f"  情绪一致性: {factors.get('sentiment_consistency_level', '未知')}")
    
    # 市场验证
    print("\n【市场验证度】")
    print(f"  股价表现: {factors.get('price_change_pct', 0):.2f}%")
    print(f"  验证度: {factors.get('market_validation_score', 0):.1f} ({factors.get('market_validation_level', '未知')})")
    print(f"  市场状态: {factors.get('market_status', '未知')}")
    print(f"  预期状态: {factors.get('expectation_status', '未知')}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
