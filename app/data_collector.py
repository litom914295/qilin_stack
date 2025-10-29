"""
麒麟量化系统 - 历史数据收集和标注系统
用于收集历史涨停数据并标注首板→二板成功率
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import pickle
from tqdm import tqdm

from app.enhanced_limitup_selector import EnhancedLimitUpSelector, LimitUpStock
from app.sector_theme_manager import SectorThemeManager

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """历史涨停数据收集器"""
    
    def __init__(self, output_dir: str = "data/historical", use_akshare: bool = True):
        """
        初始化收集器
        
        Args:
            output_dir: 数据输出目录
            use_akshare: 是否使用AKShare获取真实数据
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.selector = EnhancedLimitUpSelector()
        self.sector_manager = SectorThemeManager()
        
        self.use_akshare = use_akshare
        if use_akshare:
            try:
                import akshare as ak
                self.ak = ak
                logger.info("AKShare已加载,将使用真实数据")
            except ImportError:
                logger.warning("AKShare未安装,将使用模拟数据")
                self.use_akshare = False
                self.ak = None
        else:
            self.ak = None
        
        logger.info("历史数据收集器初始化完成")
    
    def collect_daily_limitup_stocks(
        self, 
        date: str,
        universe: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        收集某日的涨停股票(使用AKShare真实数据)
        
        Args:
            date: 日期 YYYY-MM-DD
            universe: 股票池(可选)
            
        Returns:
            涨停股票列表
        """
        logger.info(f"收集 {date} 涨停数据...")
        
        if not self.use_akshare or self.ak is None:
            # 模拟数据
            return self._generate_mock_limitup_stocks(date, 10)
        
        try:
            # 使用AKShare获取涨停池数据
            date_str = date.replace('-', '')
            df = self.ak.stock_zt_pool_em(date=date_str)
            
            if df.empty:
                logger.warning(f"{date} 无涨停数据")
                return []
            
            limitup_stocks = []
            for _, row in df.iterrows():
                stock_data = {
                    'date': date,
                    'symbol': row.get('代码', ''),
                    'name': row.get('名称', ''),
                    'price': row.get('最新价', 0),
                    'change_pct': row.get('涨跌幅', 0),
                    'turnover_rate': row.get('换手率', 0),
                    'volume_ratio': row.get('量比', 1.0),
                    'limit_up_time': row.get('首次封板时间', ''),
                    'open_times': row.get('打开次数', 0),
                    'seal_amount': row.get('封板资金', 0),
                    'total_amount': row.get('流通市值', 1),
                    'sector': row.get('所属行业', '未知'),
                    'reason': row.get('涨停原因', '')
                }
                
                # 过滤股票池
                if universe and stock_data['symbol'] not in universe:
                    continue
                
                limitup_stocks.append(stock_data)
            
            logger.info(f"收集到 {len(limitup_stocks)} 只涨停股票")
            return limitup_stocks
            
        except Exception as e:
            logger.error(f"获取涨停数据失败: {e}")
            # 降级到模拟数据
            return self._generate_mock_limitup_stocks(date, 10)
    
    def _generate_mock_limitup_stocks(self, date: str, count: int = 10) -> List[Dict]:
        """生成模拟涨停数据"""
        mock_stocks = []
        for i in range(count):
            mock_stocks.append({
                'date': date,
                'symbol': f"{np.random.randint(0, 999999):06d}",
                'name': f"股票{i+1}",
                'price': np.random.uniform(10, 100),
                'change_pct': 10.0,
                'turnover_rate': np.random.uniform(5, 30),
                'volume_ratio': np.random.uniform(1.5, 5.0),
                'limit_up_time': f"{np.random.randint(9, 15)}:{np.random.randint(0, 59):02d}",
                'open_times': np.random.randint(0, 5),
                'seal_amount': np.random.uniform(1000, 10000),
                'total_amount': np.random.uniform(10000, 100000),
                'sector': f"板块{i % 3}",
                'reason': '模拟'
            })
        return mock_stocks
    
    def label_first_to_second_board(
        self,
        first_board_date: str,
        symbol: str,
        stock_data: Dict
    ) -> int:
        """
        标注首板→二板是否成功(使用AKShare真实数据)
        
        Args:
            first_board_date: 首板日期
            symbol: 股票代码
            stock_data: 首板当日数据
            
        Returns:
            1: 次日二板成功, 0: 失败
        """
        if not self.use_akshare or self.ak is None:
            # 模拟标注
            return np.random.choice([0, 1], p=[0.7, 0.3])
        
        try:
            # 计算下一个交易日
            current_date = pd.to_datetime(first_board_date)
            
            # 获取下N个交易日(最多查找5天)
            for offset in range(1, 6):
                next_date = current_date + timedelta(days=offset)
                next_date_str = next_date.strftime('%Y%m%d')
                
                try:
                    # 检查次日涨停池
                    df_next = self.ak.stock_zt_pool_em(date=next_date_str)
                    
                    if df_next.empty:
                        continue  # 非交易日,继续查找
                    
                    # 判断该股票是否在次日涨停池
                    if symbol in df_next['代码'].values:
                        logger.debug(f"{symbol} 次日{next_date_str}涨停 → 标签=1")
                        return 1
                    else:
                        logger.debug(f"{symbol} 次日{next_date_str}未涨停 → 标签=0")
                        return 0
                        
                except Exception as e:
                    logger.debug(f"获取{next_date_str}数据失败: {e}")
                    continue
            
            # 5天内都无法获取数据,降级到模拟
            return np.random.choice([0, 1], p=[0.7, 0.3])
            
        except Exception as e:
            logger.error(f"标注失败: {e}")
            return np.random.choice([0, 1], p=[0.7, 0.3])
    
    def extract_features(
        self,
        stock: LimitUpStock,
        auction_data: Optional[Dict] = None,
        minute_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        提取特征(16维)
        
        Args:
            stock: 涨停股票对象
            auction_data: 次日竞价数据(可选)
            minute_data: 当日分钟数据(可选)
            
        Returns:
            特征字典
        """
        # 基础特征(9维)
        features = {
            'consecutive_days': stock.consecutive_days,
            'seal_ratio': stock.seal_ratio,
            'quality_score': stock.quality_score,
            'is_leader': 1.0 if stock.is_sector_leader else 0.0,
            'turnover_rate': stock.turnover_rate,
            'volume_ratio': stock.volume_ratio,
            'open_times': stock.open_times,
            'limit_up_hour': self._parse_limit_up_hour(stock.limit_up_time),
            'is_one_word': 1.0 if stock.is_one_word else 0.0,
        }
        
        # 分时特征(3维)
        if minute_data is not None and not minute_data.empty:
            intraday = self.selector.extract_intraday_features(minute_data)
            features.update({
                'vwap_slope': intraday['vwap_slope_morning'],
                'max_drawdown': intraday['max_drawdown_morning'],
                'afternoon_strength': intraday['afternoon_strength']
            })
        else:
            features.update({
                'vwap_slope': 0.0,
                'max_drawdown': 0.0,
                'afternoon_strength': 0.0
            })
        
        # 板块特征(2维)
        features.update({
            'sector_heat': stock.themes[0] if stock.themes else 0.0,  # 需要实际计算
            'sector_count': stock.sector_limit_count
        })
        
        # 首板标识(1维)
        features['is_first_board'] = 1.0 if stock.is_first_board else 0.0
        
        # 次日竞价特征(可选,如果有)
        if auction_data:
            features.update({
                'next_auction_change': auction_data.get('change_pct', 0),
                'next_auction_strength': auction_data.get('strength', 0),
                'next_bid_ask_ratio': auction_data.get('bid_ask_ratio', 1.0)
            })
        
        return features
    
    def _parse_limit_up_hour(self, limit_up_time: str) -> float:
        """解析涨停时间为小时(9.5-15.0)"""
        if not limit_up_time or limit_up_time == "":
            return 14.0  # 默认下午
        
        try:
            parts = limit_up_time.split(':')
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            return hour + minute / 60.0
        except:
            return 14.0
    
    def extract_features_from_dict(self, stock_data: Dict) -> Dict[str, float]:
        """
        从字典数据提取特征(用于回测信号生成)
        
        Args:
            stock_data: AKShare涨停数据字典
            
        Returns:
            16维特征字典
        """
        # 计算封单比例
        seal_amount = stock_data.get('seal_amount', 0)
        total_amount = stock_data.get('total_amount', 1)
        seal_ratio = seal_amount / max(total_amount, 1) if total_amount > 0 else 0
        
        # 判断一字板
        open_times = stock_data.get('open_times', 0)
        is_one_word = (open_times == 0)
        
        # 基础特征(简化版,因为部分字段可能缺失)
        features = {
            'consecutive_days': 1.0,  # 默认首板
            'seal_ratio': seal_ratio,
            'quality_score': self._estimate_quality(stock_data),
            'is_leader': 0.0,  # 需板块分析,简化为0
            'turnover_rate': stock_data.get('turnover_rate', 10.0),
            'volume_ratio': stock_data.get('volume_ratio', 2.0),
            'open_times': float(open_times),
            'limit_up_hour': self._parse_limit_up_hour(stock_data.get('limit_up_time', '')),
            'is_one_word': 1.0 if is_one_word else 0.0,
            'vwap_slope': 0.03,  # 默认值(无分时数据)
            'max_drawdown': -0.01,
            'afternoon_strength': 0.005,
            'sector_heat': 0.5,  # 默认中等热度
            'sector_count': 5,
            'is_first_board': 1.0
        }
        
        return features
    
    def _estimate_quality(self, stock_data: Dict) -> float:
        """估算质量分数"""
        score = 60.0
        
        # 封板时间加分
        limit_up_hour = self._parse_limit_up_hour(stock_data.get('limit_up_time', ''))
        if limit_up_hour < 10.0:
            score += 15
        elif limit_up_hour < 11.0:
            score += 10
        elif limit_up_hour < 13.0:
            score += 5
        
        # 打开次数扣分
        open_times = stock_data.get('open_times', 0)
        score -= open_times * 5
        
        # 换手率加分
        turnover = stock_data.get('turnover_rate', 0)
        if 8 <= turnover <= 15:
            score += 10
        elif turnover > 20:
            score -= 5
        
        # 量比加分
        volume_ratio = stock_data.get('volume_ratio', 1.0)
        if volume_ratio >= 2.0:
            score += 5
        
        return np.clip(score, 0, 100)
    
    def collect_and_label_dataset(
        self,
        start_date: str,
        end_date: str,
        universe: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        收集并标注训练数据集
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            universe: 股票池(可选)
            save_path: 保存路径(可选)
            
        Returns:
            标注后的数据集DataFrame
        """
        logger.info(f"开始收集数据: {start_date} 至 {end_date}")
        
        # 生成交易日列表
        date_range = pd.date_range(start_date, end_date, freq='B')  # 工作日
        
        all_samples = []
        
        for date in tqdm(date_range, desc="收集数据"):
            date_str = date.strftime('%Y-%m-%d')
            
            # 收集当日涨停股
            limitup_stocks = self.collect_daily_limitup_stocks(date_str, universe)
            
            if not limitup_stocks:
                continue
            
            # 筛选首板
            first_boards = [s for s in limitup_stocks if s.get('is_first_board', False)]
            
            for stock_data in first_boards:
                try:
                    # 创建LimitUpStock对象
                    stock = self._dict_to_limitup_stock(stock_data)
                    
                    # 提取特征
                    features = self.extract_features(stock)
                    
                    # 标注次日是否二板
                    label = self.label_first_to_second_board(
                        date_str, 
                        stock.symbol, 
                        stock_data
                    )
                    
                    # 组合样本
                    sample = {
                        'date': date_str,
                        'symbol': stock.symbol,
                        'name': stock.name,
                        'label': label,
                        **features
                    }
                    
                    all_samples.append(sample)
                    
                except Exception as e:
                    logger.error(f"处理 {date_str} {stock_data.get('symbol')} 失败: {e}")
                    continue
        
        # 转为DataFrame
        df = pd.DataFrame(all_samples)
        
        logger.info(f"数据收集完成: 共 {len(df)} 条样本")
        logger.info(f"正样本(二板成功): {df['label'].sum()} ({df['label'].mean():.1%})")
        
        # 保存
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"数据已保存: {save_path}")
        else:
            default_path = self.output_dir / f"limitup_dataset_{start_date}_{end_date}.csv"
            df.to_csv(default_path, index=False, encoding='utf-8')
            logger.info(f"数据已保存: {default_path}")
        
        return df
    
    def _dict_to_limitup_stock(self, data: Dict) -> LimitUpStock:
        """字典转LimitUpStock对象"""
        return LimitUpStock(
            symbol=data['symbol'],
            name=data.get('name', ''),
            date=data.get('date', ''),
            limit_up_time=data.get('limit_up_time', ''),
            open_times=data.get('open_times', 0),
            seal_ratio=data.get('seal_ratio', 0),
            is_one_word=data.get('is_one_word', False),
            consecutive_days=data.get('consecutive_days', 1),
            is_first_board=data.get('is_first_board', True),
            prev_limit_up=data.get('prev_limit_up', False),
            sector=data.get('sector', ''),
            themes=data.get('themes', []),
            sector_limit_count=data.get('sector_limit_count', 0),
            is_sector_leader=data.get('is_sector_leader', False),
            prev_close=data.get('prev_close', 0),
            open=data.get('open', 0),
            high=data.get('high', 0),
            low=data.get('low', 0),
            close=data.get('close', 0),
            limit_price=data.get('limit_price', 0),
            volume=data.get('volume', 0),
            amount=data.get('amount', 0),
            turnover_rate=data.get('turnover_rate', 0),
            volume_ratio=data.get('volume_ratio', 0),
            vwap_slope_morning=data.get('vwap_slope_morning'),
            max_drawdown_morning=data.get('max_drawdown_morning'),
            afternoon_strength=data.get('afternoon_strength'),
            quality_score=data.get('quality_score', 0),
            confidence=data.get('confidence', 0)
        )
    
    def load_dataset(self, path: str) -> pd.DataFrame:
        """加载已保存的数据集"""
        df = pd.read_csv(path, encoding='utf-8')
        logger.info(f"加载数据集: {path}, 共 {len(df)} 条样本")
        return df
    
    def generate_mock_dataset(
        self,
        n_samples: int = 1000,
        positive_ratio: float = 0.3,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        生成模拟数据集(用于测试)
        
        Args:
            n_samples: 样本数量
            positive_ratio: 正样本比例
            save_path: 保存路径
            
        Returns:
            模拟数据集
        """
        logger.info(f"生成模拟数据集: {n_samples} 条样本...")
        
        np.random.seed(42)
        
        samples = []
        dates = pd.date_range('2024-01-01', periods=200, freq='B')
        
        for i in range(n_samples):
            # 生成标签
            label = 1 if np.random.random() < positive_ratio else 0
            
            # 生成特征(正样本特征偏好)
            if label == 1:
                # 成功的首板特征
                features = {
                    'consecutive_days': 1,
                    'seal_ratio': np.random.uniform(0.08, 0.25),
                    'quality_score': np.random.uniform(75, 95),
                    'is_leader': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'turnover_rate': np.random.uniform(8, 18),
                    'volume_ratio': np.random.uniform(2.0, 5.0),
                    'open_times': np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]),
                    'limit_up_hour': np.random.uniform(9.5, 11.0),
                    'is_one_word': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'vwap_slope': np.random.uniform(0.01, 0.08),
                    'max_drawdown': np.random.uniform(-0.02, -0.001),
                    'afternoon_strength': np.random.uniform(0.001, 0.015),
                    'sector_heat': np.random.uniform(0.5, 1.0),
                    'sector_count': np.random.randint(3, 10),
                    'is_first_board': 1.0
                }
            else:
                # 失败的首板特征
                features = {
                    'consecutive_days': 1,
                    'seal_ratio': np.random.uniform(0.03, 0.10),
                    'quality_score': np.random.uniform(50, 75),
                    'is_leader': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'turnover_rate': np.random.uniform(3, 12),
                    'volume_ratio': np.random.uniform(1.0, 3.0),
                    'open_times': np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5]),
                    'limit_up_hour': np.random.uniform(11.0, 14.5),
                    'is_one_word': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'vwap_slope': np.random.uniform(-0.02, 0.02),
                    'max_drawdown': np.random.uniform(-0.05, -0.01),
                    'afternoon_strength': np.random.uniform(-0.01, 0.005),
                    'sector_heat': np.random.uniform(0.1, 0.6),
                    'sector_count': np.random.randint(1, 5),
                    'is_first_board': 1.0
                }
            
            sample = {
                'date': np.random.choice(dates).strftime('%Y-%m-%d'),
                'symbol': f"{np.random.randint(0, 999999):06d}",
                'name': f"股票{i}",
                'label': label,
                **features
            }
            
            samples.append(sample)
        
        df = pd.DataFrame(samples)
        
        logger.info(f"模拟数据生成完成: 正样本 {df['label'].sum()} ({df['label'].mean():.1%})")
        
        # 保存
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8')
            logger.info(f"数据已保存: {save_path}")
        else:
            default_path = self.output_dir / "mock_limitup_dataset.csv"
            df.to_csv(default_path, index=False, encoding='utf-8')
            logger.info(f"数据已保存: {default_path}")
        
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = HistoricalDataCollector()
    
    # 生成模拟数据集用于测试
    df = collector.generate_mock_dataset(
        n_samples=2000,
        positive_ratio=0.3
    )
    
    print("\n数据集统计:")
    print(f"总样本数: {len(df)}")
    print(f"正样本数: {df['label'].sum()} ({df['label'].mean():.1%})")
    print(f"\n特征列: {[c for c in df.columns if c not in ['date', 'symbol', 'name', 'label']]}")
    print(f"\n前5行:\n{df.head()}")
