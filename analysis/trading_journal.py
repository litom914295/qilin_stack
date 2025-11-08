"""
持仓日志和复盘系统
记录每笔交易的完整流程和决策依据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class TradeRecord:
    """交易记录"""
    # 基础信息
    trade_id: str
    symbol: str
    name: str
    trade_date: str
    
    # T日信息
    t_day_date: str
    t_day_close: float
    seal_strength: float
    prediction_score: float
    
    # T+1竞价信息
    t1_date: str
    t1_auction_price: float
    t1_auction_volume: int
    t1_auction_strength: str
    t1_buy_decision: str
    t1_buy_price: float
    t1_buy_volume: int
    t1_buy_amount: float
    t1_close_price: float
    t1_return: float
    
    # T+2卖出信息
    t2_date: str
    t2_open_price: float
    t2_sell_decision: str
    t2_sell_price: float
    t2_sell_volume: int
    t2_sell_amount: float
    
    # 收益信息
    total_return: float
    total_profit: float
    profit_rate: float
    hold_days: int
    
    # 决策依据
    buy_reason: str
    sell_reason: str
    
    # 风控信息
    position_size: float
    kelly_fraction: float
    risk_level: str
    
    # 市场环境
    market_condition: str
    index_change: float
    
    # 备注
    notes: str
    tags: str  # 逗号分隔的标签


class TradingJournal:
    """
    交易日志系统
    
    功能：
    1. 记录每笔交易的完整流程
    2. 存储决策依据和市场环境
    3. 提供查询和统计功能
    4. 生成复盘报告
    """
    
    def __init__(self, db_path: str = "trading_journal.db"):
        """
        初始化交易日志
        
        Parameters:
        -----------
        db_path: str
            数据库路径
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建交易记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                trade_date TEXT,
                t_day_date TEXT,
                t_day_close REAL,
                seal_strength REAL,
                prediction_score REAL,
                t1_date TEXT,
                t1_auction_price REAL,
                t1_auction_volume INTEGER,
                t1_auction_strength TEXT,
                t1_buy_decision TEXT,
                t1_buy_price REAL,
                t1_buy_volume INTEGER,
                t1_buy_amount REAL,
                t1_close_price REAL,
                t1_return REAL,
                t2_date TEXT,
                t2_open_price REAL,
                t2_sell_decision TEXT,
                t2_sell_price REAL,
                t2_sell_volume INTEGER,
                t2_sell_amount REAL,
                total_return REAL,
                total_profit REAL,
                profit_rate REAL,
                hold_days INTEGER,
                buy_reason TEXT,
                sell_reason TEXT,
                position_size REAL,
                kelly_fraction REAL,
                risk_level TEXT,
                market_condition TEXT,
                index_change REAL,
                notes TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_date ON trades(trade_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_profit_rate ON trades(profit_rate)')
        
        conn.commit()
        conn.close()
        
        print(f"✅ 交易日志数据库初始化完成: {self.db_path}")
    
    def add_trade(self, trade: TradeRecord) -> bool:
        """
        添加交易记录
        
        Parameters:
        -----------
        trade: TradeRecord
            交易记录
            
        Returns:
        --------
        bool: 是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 转换为字典
            trade_dict = asdict(trade)
            
            # 插入数据
            columns = ', '.join(trade_dict.keys())
            placeholders = ', '.join(['?'] * len(trade_dict))
            sql = f'INSERT INTO trades ({columns}) VALUES ({placeholders})'
            
            cursor.execute(sql, list(trade_dict.values()))
            conn.commit()
            conn.close()
            
            print(f"✅ 交易记录已保存: {trade.symbol} ({trade.trade_id})")
            return True
        
        except Exception as e:
            print(f"❌ 保存交易记录失败: {e}")
            return False
    
    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """获取单个交易记录"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT * FROM trades WHERE trade_id = ?',
            conn,
            params=(trade_id,)
        )
        conn.close()
        
        if df.empty:
            return None
        
        return TradeRecord(**df.iloc[0].to_dict())
    
    def query_trades(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    symbol: Optional[str] = None,
                    profit_only: bool = False) -> pd.DataFrame:
        """
        查询交易记录
        
        Parameters:
        -----------
        start_date: str
            开始日期
        end_date: str
            结束日期
        symbol: str
            股票代码
        profit_only: bool
            仅查询盈利交易
            
        Returns:
        --------
        DataFrame: 查易记录
        """
        conn = sqlite3.connect(self.db_path)
        
        # 构建查询条件
        conditions = []
        params = []
        
        if start_date:
            conditions.append('trade_date >= ?')
            params.append(start_date)
        
        if end_date:
            conditions.append('trade_date <= ?')
            params.append(end_date)
        
        if symbol:
            conditions.append('symbol = ?')
            params.append(symbol)
        
        if profit_only:
            conditions.append('profit_rate > 0')
        
        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        sql = f'SELECT * FROM trades WHERE {where_clause} ORDER BY trade_date DESC'
        
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        
        return df
    
    def generate_review_report(self,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict:
        """
        生成复盘报告
        
        Parameters:
        -----------
        start_date: str
            开始日期
        end_date: str
            结束日期
            
        Returns:
        --------
        Dict: 复盘报告
        """
        df = self.query_trades(start_date=start_date, end_date=end_date)
        
        if df.empty:
            return {'error': '无交易记录'}
        
        report = {}
        
        # 1. 基础统计
        report['basic_stats'] = {
            '交易总数': len(df),
            '盈利笔数': len(df[df['profit_rate'] > 0]),
            '亏损笔数': len(df[df['profit_rate'] <= 0]),
            '胜率': f"{len(df[df['profit_rate'] > 0]) / len(df) * 100:.2f}%",
            '平均收益率': f"{df['profit_rate'].mean():.2f}%",
            '累计收益': f"¥{df['total_profit'].sum():,.0f}",
            '平均持仓天数': f"{df['hold_days'].mean():.1f}天",
        }
        
        # 2. 收益分布
        report['return_distribution'] = {
            '最大单笔收益': f"{df['profit_rate'].max():.2f}%",
            '最大单笔亏损': f"{df['profit_rate'].min():.2f}%",
            '收益标准差': f"{df['profit_rate'].std():.2f}%",
            '盈亏比': f"{abs(df[df['profit_rate'] > 0]['profit_rate'].mean() / df[df['profit_rate'] <= 0]['profit_rate'].mean()):.2f}" if len(df[df['profit_rate'] <= 0]) > 0 else 'N/A',
        }
        
        # 3. 按竞价强度分析
        auction_analysis = df.groupby('t1_auction_strength').agg({
            'trade_id': 'count',
            'profit_rate': ['mean', 'std'],
            'total_profit': 'sum'
        }).round(2)
        report['auction_strength_analysis'] = auction_analysis.to_dict()
        
        # 4. 按市场环境分析
        market_analysis = df.groupby('market_condition').agg({
            'trade_id': 'count',
            'profit_rate': ['mean', 'std'],
            'total_profit': 'sum'
        }).round(2)
        report['market_condition_analysis'] = market_analysis.to_dict()
        
        # 5. 成功案例TOP5
        top_trades = df.nlargest(5, 'profit_rate')[['symbol', 'name', 'profit_rate', 'total_profit', 'buy_reason']]
        report['top_trades'] = top_trades.to_dict('records')
        
        # 6. 失败案例TOP5
        worst_trades = df.nsmallest(5, 'profit_rate')[['symbol', 'name', 'profit_rate', 'total_profit', 'sell_reason']]
        report['worst_trades'] = worst_trades.to_dict('records')
        
        # 7. 月度收益
        df['month'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m')
        monthly_profit = df.groupby('month')['total_profit'].sum().to_dict()
        report['monthly_profit'] = monthly_profit
        
        return report
    
    def print_report(self, report: Dict):
        """打印复盘报告"""
        print(f"\n{'='*100}")
        print(f"{'交易复盘报告':^96}")
        print(f"{'='*100}\n")
        
        # 基础统计
        print("【基础统计】")
        for key, value in report['basic_stats'].items():
            print(f"  {key}: {value}")
        
        # 收益分布
        print(f"\n【收益分布】")
        for key, value in report['return_distribution'].items():
            print(f"  {key}: {value}")
        
        # 成功案例
        print(f"\n【成功案例 TOP5】")
        for i, trade in enumerate(report['top_trades'], 1):
            print(f"  {i}. {trade['symbol']} ({trade['name']}): {trade['profit_rate']:.2f}% (¥{trade['total_profit']:,.0f})")
            print(f"     原因: {trade['buy_reason']}")
        
        # 失败案例
        print(f"\n【失败案例 TOP5】")
        for i, trade in enumerate(report['worst_trades'], 1):
            print(f"  {i}. {trade['symbol']} ({trade['name']}): {trade['profit_rate']:.2f}% (¥{trade['total_profit']:,.0f})")
            print(f"     原因: {trade['sell_reason']}")
        
        print(f"\n{'='*100}\n")
    
    def export_to_excel(self,
                       output_path: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None):
        """导出交易记录到Excel"""
        df = self.query_trades(start_date=start_date, end_date=end_date)
        
        if df.empty:
            print("无交易记录可导出")
            return
        
        # 创建Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet1: 交易明细
            df.to_excel(writer, sheet_name='交易明细', index=False)
            
            # Sheet2: 统计分析
            report = self.generate_review_report(start_date, end_date)
            stats_df = pd.DataFrame({
                '指标': list(report['basic_stats'].keys()) + list(report['return_distribution'].keys()),
                '值': list(report['basic_stats'].values()) + list(report['return_distribution'].values())
            })
            stats_df.to_excel(writer, sheet_name='统计分析', index=False)
        
        print(f"✅ 交易记录已导出到: {output_path}")


def create_sample_trade() -> TradeRecord:
    """创建示例交易记录"""
    trade_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return TradeRecord(
        trade_id=trade_id,
        symbol="000001.SZ",
        name="平安银行",
        trade_date="2024-11-01",
        
        # T日
        t_day_date="2024-10-30",
        t_day_close=11.50,
        seal_strength=7.5,
        prediction_score=0.85,
        
        # T+1
        t1_date="2024-10-31",
        t1_auction_price=11.85,
        t1_auction_volume=5000,
        t1_auction_strength="超强",
        t1_buy_decision="竞价买入",
        t1_buy_price=11.88,
        t1_buy_volume=1000,
        t1_buy_amount=11880,
        t1_close_price=12.50,
        t1_return=5.22,
        
        # T+2
        t2_date="2024-11-01",
        t2_open_price=12.68,
        t2_sell_decision="高开卖60%",
        t2_sell_price=12.65,
        t2_sell_volume=600,
        t2_sell_amount=7590,
        
        # 收益
        total_return=6.48,
        total_profit=769.8,
        profit_rate=6.48,
        hold_days=2,
        
        # 决策
        buy_reason="T日涨停封单强度高，预测分数0.85，T+1竞价超强",
        sell_reason="T+1涨停，T+2高开1.4%，兑现60%利润",
        
        # 风控
        position_size=0.05,
        kelly_fraction=0.08,
        risk_level="中",
        
        # 市场
        market_condition="正常",
        index_change=0.5,
        
        notes="首次使用新策略，效果良好",
        tags="涨停,高开,盈利"
    )


# 使用示例
if __name__ == "__main__":
    # 创建交易日志
    journal = TradingJournal("trading_journal.db")
    
    # 添加示例交易
    print("添加示例交易记录...")
    for i in range(10):
        trade = create_sample_trade()
        trade.trade_id = f"T202411{i:02d}"
        trade.symbol = f"{i:06d}.SZ"
        trade.profit_rate = np.random.uniform(-5, 15)
        trade.total_profit = trade.profit_rate * 100
        journal.add_trade(trade)
    
    # 查询交易
    print("\n查询所有交易...")
    trades_df = journal.query_trades()
    print(f"共查询到 {len(trades_df)} 条交易记录")
    
    # 生成复盘报告
    print("\n生成复盘报告...")
    report = journal.generate_review_report()
    journal.print_report(report)
    
    # 导出到Excel
    journal.export_to_excel("trading_journal.xlsx")
    
    print("\n✅ 交易日志系统测试完成！")
