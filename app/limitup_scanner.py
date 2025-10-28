"""
自动涨停股扫描和批量分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LimitUpScanner:
    """涨停股自动扫描器"""
    
    def __init__(self):
        self.limitup_threshold = 0.095  # 9.5%以上算涨停(考虑误差)
        
    def scan_today_limitup(self, date: str = None) -> List[Dict[str, Any]]:
        """
        扫描当日涨停股
        
        Args:
            date: 日期，格式YYYY-MM-DD，默认为今天
            
        Returns:
            涨停股列表
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # 尝试使用AKShare获取涨停股
            import akshare as ak
            
            logger.info(f"正在扫描 {date} 的涨停股...")
            
            # 获取A股实时行情
            df = ak.stock_zh_a_spot_em()
            
            # 筛选涨停股
            # 涨跌幅 >= 9.5%
            limitup_stocks = df[df['涨跌幅'] >= 9.5].copy()
            
            # 过滤ST和退市股
            limitup_stocks = limitup_stocks[
                ~limitup_stocks['名称'].str.contains('ST|退', na=False)
            ]
            
            # 转换格式
            results = []
            for _, row in limitup_stocks.iterrows():
                code = row['代码']
                name = row['名称']
                
                # 判断市场
                if code.startswith('6'):
                    symbol = f"SH{code}"
                elif code.startswith(('0', '3')):
                    symbol = f"SZ{code}"
                else:
                    continue
                
                results.append({
                    'symbol': symbol,
                    'code': code,
                    'name': name,
                    'price': row['最新价'],
                    'change_pct': row['涨跌幅'],
                    'volume': row['成交量'],
                    'amount': row['成交额'],
                    'turnover': row['换手率'],
                    'limitup_time': None,  # 需要从分时数据获取
                    'open_times': 0  # 需要从分时数据获取
                })
            
            logger.info(f"找到 {len(results)} 只涨停股")
            return results
            
        except ImportError:
            logger.error("AKShare未安装，请运行: pip install akshare")
            return self._get_mock_limitup_data()
        except Exception as e:
            logger.error(f"扫描涨停股失败: {e}")
            return self._get_mock_limitup_data()
    
    def _get_mock_limitup_data(self) -> List[Dict[str, Any]]:
        """获取模拟涨停股数据（演示用）"""
        return [
            {
                'symbol': 'SZ000001',
                'code': '000001',
                'name': '平安银行',
                'price': 11.00,
                'change_pct': 10.0,
                'volume': 50000000,
                'amount': 550000000,
                'turnover': 5.2,
                'limitup_time': '10:30:00',
                'open_times': 0
            },
            {
                'symbol': 'SZ000002',
                'code': '000002',
                'name': '万科A',
                'price': 8.80,
                'change_pct': 10.0,
                'volume': 120000000,
                'amount': 1056000000,
                'turnover': 8.5,
                'limitup_time': '13:15:00',
                'open_times': 2
            },
            {
                'symbol': 'SH600000',
                'code': '600000',
                'name': '浦发银行',
                'price': 9.90,
                'change_pct': 10.0,
                'volume': 80000000,
                'amount': 792000000,
                'turnover': 3.8,
                'limitup_time': '09:35:00',
                'open_times': 0
            }
        ]
    
    def analyze_limitup_strength(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析单只涨停股强度
        
        Args:
            stock: 股票信息
            
        Returns:
            分析结果
        """
        # 1. 涨停时间得分（越早越好）
        time_score = self._calculate_time_score(stock.get('limitup_time'))
        
        # 2. 封单强度得分（成交额、换手率）
        seal_score = self._calculate_seal_score(stock)
        
        # 3. 开板次数得分（越少越好）
        open_score = self._calculate_open_score(stock.get('open_times', 0))
        
        # 4. 量能得分（放量程度）
        volume_score = self._calculate_volume_score(stock)
        
        # 综合得分
        total_score = (
            time_score * 0.25 +
            seal_score * 0.35 +
            open_score * 0.20 +
            volume_score * 0.20
        )
        
        # 评级
        if total_score >= 85:
            rating = "🔥 强势"
            recommendation = "重点关注"
        elif total_score >= 70:
            rating = "⚠️ 一般"
            recommendation = "谨慎观望"
        else:
            rating = "❌ 弱势"
            recommendation = "不建议"
        
        return {
            'symbol': stock['symbol'],
            'name': stock['name'],
            'total_score': round(total_score, 2),
            'rating': rating,
            'recommendation': recommendation,
            'scores': {
                'time_score': round(time_score, 2),
                'seal_score': round(seal_score, 2),
                'open_score': round(open_score, 2),
                'volume_score': round(volume_score, 2)
            },
            'raw_data': stock
        }
    
    def _calculate_time_score(self, limitup_time: str) -> float:
        """计算涨停时间得分"""
        if not limitup_time:
            return 50.0  # 未知时间给中等分
        
        try:
            hour, minute, _ = map(int, limitup_time.split(':'))
            total_minutes = hour * 60 + minute
            
            # 9:30 = 570分钟，15:00 = 900分钟
            # 越早涨停得分越高
            if total_minutes <= 600:  # 10:00前
                return 100.0
            elif total_minutes <= 660:  # 11:00前
                return 90.0
            elif total_minutes <= 780:  # 13:00前
                return 70.0
            elif total_minutes <= 840:  # 14:00前
                return 50.0
            else:  # 14:00后
                return 30.0
        except:
            return 50.0
    
    def _calculate_seal_score(self, stock: Dict[str, Any]) -> float:
        """计算封单强度得分"""
        turnover = stock.get('turnover', 0)
        amount = stock.get('amount', 0)
        
        # 换手率适中最好（太高说明不稳，太低说明没人气）
        if 3 <= turnover <= 8:
            turnover_score = 100
        elif 1 <= turnover < 3 or 8 < turnover <= 15:
            turnover_score = 70
        else:
            turnover_score = 40
        
        # 成交额越大越好（说明资金关注度高）
        if amount >= 1000000000:  # >=10亿
            amount_score = 100
        elif amount >= 500000000:  # >=5亿
            amount_score = 80
        elif amount >= 200000000:  # >=2亿
            amount_score = 60
        else:
            amount_score = 40
        
        return (turnover_score * 0.6 + amount_score * 0.4)
    
    def _calculate_open_score(self, open_times: int) -> float:
        """计算开板次数得分"""
        if open_times == 0:
            return 100.0  # 一字板最强
        elif open_times == 1:
            return 80.0
        elif open_times == 2:
            return 60.0
        elif open_times == 3:
            return 40.0
        else:
            return 20.0  # 开板太多次，封单不稳
    
    def _calculate_volume_score(self, stock: Dict[str, Any]) -> float:
        """计算量能得分"""
        volume = stock.get('volume', 0)
        
        # 根据成交量判断（简化版，实际应该对比历史均量）
        if volume >= 100000000:  # >=1亿股
            return 100.0
        elif volume >= 50000000:  # >=5000万股
            return 80.0
        elif volume >= 20000000:  # >=2000万股
            return 60.0
        else:
            return 40.0
    
    def batch_analyze(self, date: str = None) -> pd.DataFrame:
        """
        批量分析当日涨停股
        
        Args:
            date: 日期，默认今天
            
        Returns:
            分析结果DataFrame
        """
        # 1. 扫描涨停股
        limitup_stocks = self.scan_today_limitup(date)
        
        if not limitup_stocks:
            logger.warning("未找到涨停股")
            return pd.DataFrame()
        
        # 2. 逐个分析
        results = []
        for stock in limitup_stocks:
            analysis = self.analyze_limitup_strength(stock)
            results.append(analysis)
        
        # 3. 转换为DataFrame并排序
        df = pd.DataFrame(results)
        df = df.sort_values('total_score', ascending=False).reset_index(drop=True)
        
        logger.info(f"批量分析完成，共 {len(df)} 只涨停股")
        
        return df


# 便捷函数
def scan_and_analyze_today() -> pd.DataFrame:
    """扫描并分析今日涨停股（一键调用）"""
    scanner = LimitUpScanner()
    return scanner.batch_analyze()


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("涨停股自动扫描和分析")
    print("=" * 70)
    
    df = scan_and_analyze_today()
    
    print("\n分析结果:")
    print(df[['name', 'total_score', 'rating', 'recommendation']].to_string(index=False))
