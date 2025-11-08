"""
Phase 4高级功能模块
集成模拟交易、策略回测、数据导出功能
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import io

from .color_scheme import Colors, Emojis, get_profit_color, get_profit_emoji
from .loading_cache import LoadingSpinner, show_success_animation, show_error_animation


# ==================== 模拟交易系统 ====================

class SimulatedTrading:
    """模拟交易系统"""
    
    def __init__(self):
        """初始化模拟交易系统"""
        if 'simulated_positions' not in st.session_state:
            st.session_state.simulated_positions = []
        if 'simulated_history' not in st.session_state:
            st.session_state.simulated_history = []
        if 'simulated_capital' not in st.session_state:
            st.session_state.simulated_capital = 100000  # 初始资金10万
    
    def buy(self, symbol: str, price: float, quantity: int, date: str = None) -> Dict:
        """
        模拟买入
        
        Args:
            symbol: 股票代码
            price: 买入价格
            quantity: 买入数量
            date: 交易日期
            
        Returns:
            交易记录字典
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        cost = price * quantity
        
        # 检查资金是否足够
        if cost > st.session_state.simulated_capital:
            return {
                'success': False,
                'message': f"资金不足！需要{cost:.2f}元，可用{st.session_state.simulated_capital:.2f}元"
            }
        
        # 扣除资金
        st.session_state.simulated_capital -= cost
        
        # 添加持仓
        position = {
            'symbol': symbol,
            'buy_price': price,
            'quantity': quantity,
            'buy_date': date,
            'cost': cost
        }
        st.session_state.simulated_positions.append(position)
        
        # 记录交易历史
        trade_record = {
            'type': 'buy',
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'amount': cost,
            'date': date,
            'capital_after': st.session_state.simulated_capital
        }
        st.session_state.simulated_history.append(trade_record)
        
        return {
            'success': True,
            'message': f"成功买入{symbol} {quantity}股，花费{cost:.2f}元",
            'position': position
        }
    
    def sell(self, symbol: str, price: float, quantity: int = None, date: str = None) -> Dict:
        """
        模拟卖出
        
        Args:
            symbol: 股票代码
            price: 卖出价格
            quantity: 卖出数量（None表示全部卖出）
            date: 交易日期
            
        Returns:
            交易记录字典
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 查找持仓
        position_idx = None
        for i, pos in enumerate(st.session_state.simulated_positions):
            if pos['symbol'] == symbol:
                position_idx = i
                break
        
        if position_idx is None:
            return {
                'success': False,
                'message': f"未找到{symbol}的持仓"
            }
        
        position = st.session_state.simulated_positions[position_idx]
        
        # 确定卖出数量
        sell_qty = quantity if quantity is not None else position['quantity']
        if sell_qty > position['quantity']:
            return {
                'success': False,
                'message': f"卖出数量({sell_qty})超过持仓数量({position['quantity']})"
            }
        
        # 计算收益
        revenue = price * sell_qty
        cost = position['buy_price'] * sell_qty
        profit = revenue - cost
        profit_rate = (profit / cost) * 100
        
        # 更新资金
        st.session_state.simulated_capital += revenue
        
        # 更新持仓
        if sell_qty == position['quantity']:
            # 全部卖出，移除持仓
            st.session_state.simulated_positions.pop(position_idx)
        else:
            # 部分卖出，更新持仓数量
            position['quantity'] -= sell_qty
            position['cost'] = position['buy_price'] * position['quantity']
        
        # 记录交易历史
        trade_record = {
            'type': 'sell',
            'symbol': symbol,
            'price': price,
            'quantity': sell_qty,
            'amount': revenue,
            'profit': profit,
            'profit_rate': profit_rate,
            'date': date,
            'capital_after': st.session_state.simulated_capital
        }
        st.session_state.simulated_history.append(trade_record)
        
        return {
            'success': True,
            'message': f"成功卖出{symbol} {sell_qty}股，收入{revenue:.2f}元，盈亏{profit:+.2f}元({profit_rate:+.2f}%)",
            'profit': profit,
            'profit_rate': profit_rate
        }
    
    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        if not st.session_state.simulated_positions:
            return pd.DataFrame()
        return pd.DataFrame(st.session_state.simulated_positions)
    
    def get_history(self) -> pd.DataFrame:
        """获取交易历史"""
        if not st.session_state.simulated_history:
            return pd.DataFrame()
        return pd.DataFrame(st.session_state.simulated_history)
    
    def get_statistics(self) -> Dict:
        """获取交易统计"""
        history = self.get_history()
        
        if history.empty:
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'current_capital': st.session_state.simulated_capital,
                'total_return': 0
            }
        
        sell_trades = history[history['type'] == 'sell']
        
        if sell_trades.empty:
            return {
                'total_trades': len(history),
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'current_capital': st.session_state.simulated_capital,
                'total_return': 0
            }
        
        win_trades = len(sell_trades[sell_trades['profit'] > 0])
        loss_trades = len(sell_trades[sell_trades['profit'] <= 0])
        win_rate = (win_trades / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
        
        total_profit = sell_trades['profit'].sum()
        avg_profit = sell_trades['profit'].mean()
        
        initial_capital = 100000
        total_return = ((st.session_state.simulated_capital - initial_capital) / initial_capital) * 100
        
        return {
            'total_trades': len(history),
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'current_capital': st.session_state.simulated_capital,
            'total_return': total_return
        }
    
    def reset(self):
        """重置模拟交易"""
        st.session_state.simulated_positions = []
        st.session_state.simulated_history = []
        st.session_state.simulated_capital = 100000


# ==================== 策略回测引擎 ====================

class StrategyBacktest:
    """策略回测引擎"""
    
    def __init__(self):
        """初始化回测引擎"""
        pass
    
    def backtest(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float = 100000,
        commission_rate: float = 0.001
    ) -> Dict:
        """
        执行回测
        
        Args:
            signals_df: 信号DataFrame (需包含: date, symbol, action, price列)
            initial_capital: 初始资金
            commission_rate: 手续费率
            
        Returns:
            回测结果字典
        """
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = [initial_capital]
        dates = [signals_df['date'].min()]
        
        for _, row in signals_df.iterrows():
            date = row['date']
            symbol = row['symbol']
            action = row['action']
            price = row['price']
            
            if action == 'buy' and symbol not in positions:
                # 买入
                quantity = int((capital * 0.3) / price)  # 每次买入30%资金
                if quantity > 0:
                    cost = price * quantity * (1 + commission_rate)
                    if cost <= capital:
                        capital -= cost
                        positions[symbol] = {
                            'quantity': quantity,
                            'buy_price': price,
                            'buy_date': date
                        }
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'buy',
                            'price': price,
                            'quantity': quantity,
                            'amount': -cost
                        })
            
            elif action == 'sell' and symbol in positions:
                # 卖出
                pos = positions[symbol]
                revenue = price * pos['quantity'] * (1 - commission_rate)
                capital += revenue
                
                cost = pos['buy_price'] * pos['quantity']
                profit = revenue - cost
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'price': price,
                    'quantity': pos['quantity'],
                    'amount': revenue,
                    'profit': profit,
                    'profit_rate': (profit / cost) * 100
                })
                
                del positions[symbol]
            
            # 记录权益曲线
            position_value = sum(pos['quantity'] * price for pos in positions.values())
            total_equity = capital + position_value
            equity_curve.append(total_equity)
            dates.append(date)
        
        # 计算统计指标
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty or len(trades_df[trades_df['action'] == 'sell']) == 0:
            return {
                'equity_curve': equity_curve,
                'dates': dates,
                'trades': trades,
                'statistics': {
                    'total_return': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'avg_profit': 0,
                    'max_drawdown': 0
                }
            }
        
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
        win_trades = len(sell_trades[sell_trades['profit'] > 0])
        total_trades = len(sell_trades)
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_profit = sell_trades['profit'].mean() if not sell_trades.empty else 0
        
        # 计算最大回撤
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'equity_curve': equity_curve,
            'dates': dates,
            'trades': trades,
            'statistics': {
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'avg_profit': avg_profit,
                'max_drawdown': -max_drawdown
            }
        }
    
    def plot_equity_curve(self, backtest_result: Dict) -> go.Figure:
        """绘制权益曲线"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=backtest_result['dates'],
            y=backtest_result['equity_curve'],
            mode='lines',
            name='权益曲线',
            line=dict(color=Colors.PRIMARY, width=2)
        ))
        
        fig.update_layout(
            title="策略权益曲线",
            xaxis_title="日期",
            yaxis_title="资金（元）",
            template="plotly_white",
            height=400
        )
        
        return fig


# ==================== 数据导出管理器 ====================

class ExportManager:
    """数据导出管理器"""
    
    @staticmethod
    def export_to_excel(
        data_dict: Dict[str, pd.DataFrame],
        filename: str = "limitup_report.xlsx"
    ) -> bytes:
        """
        导出多个DataFrame到Excel
        
        Args:
            data_dict: {sheet_name: DataFrame}
            filename: 文件名
            
        Returns:
            Excel文件字节流
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame) -> bytes:
        """
        导出DataFrame到CSV
        
        Args:
            df: DataFrame
            
        Returns:
            CSV文件字节流
        """
        return df.to_csv(index=False).encode('utf-8-sig')
    
    @staticmethod
    def export_to_json(data: Any, pretty: bool = True) -> bytes:
        """
        导出数据到JSON
        
        Args:
            data: 要导出的数据
            pretty: 是否格式化
            
        Returns:
            JSON文件字节流
        """
        if pretty:
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            json_str = json.dumps(data, ensure_ascii=False)
        
        return json_str.encode('utf-8')
    
    @staticmethod
    def create_report(
        candidate_df: pd.DataFrame,
        statistics: Dict,
        export_format: str = "excel"
    ) -> bytes:
        """
        创建完整报告
        
        Args:
            candidate_df: 候选股DataFrame
            statistics: 统计数据
            export_format: 导出格式 (excel/csv/json)
            
        Returns:
            报告文件字节流
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "excel":
            # Excel报告（多个sheet）
            data_dict = {
                "候选股列表": candidate_df,
                "统计信息": pd.DataFrame([statistics])
            }
            return ExportManager.export_to_excel(data_dict, f"limitup_report_{timestamp}.xlsx")
        
        elif export_format == "csv":
            # CSV报告
            return ExportManager.export_to_csv(candidate_df)
        
        elif export_format == "json":
            # JSON报告
            report = {
                "timestamp": timestamp,
                "candidates": candidate_df.to_dict(orient='records'),
                "statistics": statistics
            }
            return ExportManager.export_to_json(report)
        
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")


# ==================== 渲染函数 ====================

def render_simulated_trading(trading: SimulatedTrading):
    """渲染模拟交易界面"""
    st.markdown("### 💰 模拟交易系统")
    
    # 显示资金状态
    stats = trading.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前资金", f"¥{stats['current_capital']:,.0f}")
    with col2:
        st.metric("总收益率", f"{stats['total_return']:+.2f}%")
    with col3:
        st.metric("胜率", f"{stats['win_rate']:.1f}%")
    with col4:
        st.metric("交易次数", stats['total_trades'])
    
    # 交易操作
    st.markdown("#### 📝 交易操作")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**买入**")
        buy_symbol = st.text_input("股票代码", key="sim_buy_symbol")
        buy_price = st.number_input("买入价格", min_value=0.01, value=10.0, step=0.01, key="sim_buy_price")
        buy_qty = st.number_input("买入数量", min_value=100, value=1000, step=100, key="sim_buy_qty")
        
        if st.button(f"{Emojis.BUY} 模拟买入"):
            result = trading.buy(buy_symbol, buy_price, buy_qty)
            if result['success']:
                show_success_animation(result['message'])
            else:
                show_error_animation(result['message'])
    
    with col2:
        st.markdown("**卖出**")
        positions = trading.get_positions()
        if not positions.empty:
            sell_symbol = st.selectbox("选择持仓", positions['symbol'].tolist(), key="sim_sell_symbol")
            sell_price = st.number_input("卖出价格", min_value=0.01, value=10.0, step=0.01, key="sim_sell_price")
            
            if st.button(f"{Emojis.SELL} 模拟卖出"):
                result = trading.sell(sell_symbol, sell_price)
                if result['success']:
                    show_success_animation(result['message'])
                else:
                    show_error_animation(result['message'])
        else:
            st.info("当前无持仓")
    
    # 显示持仓
    st.markdown("#### 📊 当前持仓")
    if not positions.empty:
        st.dataframe(positions, use_container_width=True)
    else:
        st.info("暂无持仓")
    
    # 显示交易历史
    st.markdown("#### 📜 交易历史")
    history = trading.get_history()
    if not history.empty:
        st.dataframe(history.tail(10), use_container_width=True)
    else:
        st.info("暂无交易记录")
    
    # 重置按钮
    if st.button("🔄 重置模拟交易"):
        trading.reset()
        st.rerun()


def render_backtest(backtest_engine: StrategyBacktest):
    """渲染回测界面"""
    st.markdown("### 📈 策略回测")
    
    st.info("💡 回测功能需要历史信号数据。这里提供一个示例回测。")
    
    # 示例回测数据
    if st.button("运行示例回测"):
        with LoadingSpinner("正在执行回测...", Emojis.CHART):
            # 生成示例信号
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            signals = []
            
            for i, date in enumerate(dates[:50]):  # 前50天
                if i % 10 == 0:  # 每10天买入
                    signals.append({
                        'date': date,
                        'symbol': f'00000{i//10+1}',
                        'action': 'buy',
                        'price': 10 + (i % 5)
                    })
                elif i % 10 == 5:  # 5天后卖出
                    signals.append({
                        'date': date,
                        'symbol': f'00000{i//10}',
                        'action': 'sell',
                        'price': 10 + (i % 5) + 0.5
                    })
            
            signals_df = pd.DataFrame(signals)
            result = backtest_engine.backtest(signals_df)
        
        # 显示结果
        st.markdown("#### 📊 回测结果")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总收益率", f"{result['statistics']['total_return']:+.2f}%")
        with col2:
            st.metric("胜率", f"{result['statistics']['win_rate']:.1f}%")
        with col3:
            st.metric("交易次数", result['statistics']['total_trades'])
        with col4:
            st.metric("最大回撤", f"{result['statistics']['max_drawdown']:.2f}%")
        
        # 绘制权益曲线
        fig = backtest_engine.plot_equity_curve(result)
        st.plotly_chart(fig, use_container_width=True)


def render_export(candidate_df: pd.DataFrame, statistics: Dict):
    """渲染数据导出界面"""
    st.markdown("### 📤 数据导出")
    
    export_format = st.selectbox(
        "选择导出格式",
        options=["Excel", "CSV", "JSON"],
        index=0
    )
    
    if st.button(f"{Emojis.EXPORT} 导出报告"):
        with LoadingSpinner("正在生成报告...", Emojis.SAVE):
            try:
                format_map = {"Excel": "excel", "CSV": "csv", "JSON": "json"}
                file_data = ExportManager.create_report(
                    candidate_df,
                    statistics,
                    format_map[export_format]
                )
                
                filename_map = {
                    "Excel": "limitup_report.xlsx",
                    "CSV": "limitup_report.csv",
                    "JSON": "limitup_report.json"
                }
                mime_map = {
                    "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "CSV": "text/csv",
                    "JSON": "application/json"
                }
                
                st.download_button(
                    label=f"⬇️ 下载{export_format}报告",
                    data=file_data,
                    file_name=filename_map[export_format],
                    mime=mime_map[export_format]
                )
                
                show_success_animation(f"{export_format}报告生成成功！")
            
            except Exception as e:
                show_error_animation(f"导出失败: {str(e)}")


__all__ = [
    'SimulatedTrading',
    'StrategyBacktest',
    'ExportManager',
    'render_simulated_trading',
    'render_backtest',
    'render_export',
]
