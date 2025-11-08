"""
智能提示系统和一键操作按钮组
根据数据和阶段动态生成建议，提供快捷操作
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


class SmartTipSystem:
    """智能提示系统"""
    
    def __init__(self):
        """初始化智能提示系统"""
        pass
    
    def generate_tips(self, stage: str, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        根据当前阶段和数据生成智能提示
        
        Args:
            stage: 当前交易阶段
            data: 相关数据字典
            
        Returns:
            提示列表 [{'type': 'success/info/warning/error', 'message': '...'}]
        """
        tips = []
        
        if stage == "T日选股":
            tips.extend(self._generate_t_day_tips(data))
        elif stage == "T+1竞价监控":
            tips.extend(self._generate_t1_auction_tips(data))
        elif stage == "T+1盘中交易":
            tips.extend(self._generate_t1_trading_tips(data))
        elif stage == "T+2卖出决策":
            tips.extend(self._generate_t2_sell_tips(data))
        
        return tips
    
    def _generate_t_day_tips(self, data: Dict) -> List[Dict]:
        """T日选股阶段的提示"""
        tips = []
        
        limitup_count = data.get('limitup_count', 0)
        candidate_count = data.get('candidate_count', 0)
        avg_quality = data.get('avg_quality_score', 0)
        
        # 涨停数量分析
        if limitup_count > 100:
            tips.append({
                'type': 'success',
                'message': f"💡 今日涨停 {limitup_count} 只，市场情绪活跃，可适当放宽筛选条件"
            })
        elif limitup_count < 30:
            tips.append({
                'type': 'warning',
                'message': f"⚠️  今日涨停仅 {limitup_count} 只，市场情绪低迷，建议提高筛选标准"
            })
        
        # 候选池分析
        if candidate_count > 15:
            tips.append({
                'type': 'warning',
                'message': f"⚠️  候选池 {candidate_count} 只偏多，建议进一步筛选，聚焦核心标的"
            })
        elif candidate_count == 0:
            tips.append({
                'type': 'error',
                'message': "❌ 当前无候选股票，请调整筛选条件或降低质量要求"
            })
        elif 5 <= candidate_count <= 10:
            tips.append({
                'type': 'success',
                'message': f"✅ 候选池 {candidate_count} 只，数量适中，建议重点分析各标的基本面"
            })
        
        # 质量分析
        if avg_quality >= 80:
            tips.append({
                'type': 'success',
                'message': f"💯 候选股平均质量分 {avg_quality:.1f}，整体质量优秀"
            })
        elif avg_quality < 60:
            tips.append({
                'type': 'warning',
                'message': f"⚠️  候选股平均质量分 {avg_quality:.1f}，建议提高筛选标准"
            })
        
        return tips
    
    def _generate_t1_auction_tips(self, data: Dict) -> List[Dict]:
        """T+1竞价监控阶段的提示"""
        tips = []
        
        strong_count = data.get('strong_count', 0)
        weak_count = data.get('weak_count', 0)
        avg_strength = data.get('avg_strength', 0)
        monitor_count = data.get('monitor_count', 0)
        
        # 强势股分析
        if strong_count > 0:
            tips.append({
                'type': 'success',
                'message': f"🟢 {strong_count} 只候选股竞价强势（涨幅>5%），建议优先买入"
            })
        
        # 弱势股提示
        if weak_count > 0:
            tips.append({
                'type': 'error',
                'message': f"🔴 {weak_count} 只候选股竞价走弱（跌幅>5%），建议放弃"
            })
        
        # 整体强度分析
        if avg_strength > 5:
            tips.append({
                'type': 'success',
                'message': f"💪 平均竞价强度 {avg_strength:+.2f}%，市场承接力强，可积极操作"
            })
        elif avg_strength < 0:
            tips.append({
                'type': 'warning',
                'message': f"⚠️  平均竞价强度 {avg_strength:+.2f}%，市场分歧较大，建议谨慎"
            })
        
        # 监控数量提示
        if monitor_count > 0:
            tips.append({
                'type': 'info',
                'message': f"👁️ 当前监控 {monitor_count} 只股票，重点关注竞价涨幅 >5% 的标的"
            })
        
        return tips
    
    def _generate_t1_trading_tips(self, data: Dict) -> List[Dict]:
        """T+1盘中交易阶段的提示"""
        tips = []
        
        position_count = data.get('position_count', 0)
        
        if position_count > 0:
            tips.append({
                'type': 'success',
                'message': f"✅ 当前持仓 {position_count} 只，关注盘中走势和资金流向"
            })
        else:
            tips.append({
                'type': 'info',
                'message': "💡 当前无持仓，可关注盘中低吸机会或等待下一个交易日"
            })
        
        return tips
    
    def _generate_t2_sell_tips(self, data: Dict) -> List[Dict]:
        """T+2卖出决策阶段的提示"""
        tips = []
        
        profit_count = data.get('profit_count', 0)
        loss_count = data.get('loss_count', 0)
        high_profit_count = data.get('high_profit_count', 0)
        
        # 盈利分析
        if profit_count > 0:
            tips.append({
                'type': 'success',
                'message': f"💰 {profit_count} 只持仓盈利，建议根据走势适时止盈"
            })
        
        # 高盈利提示
        if high_profit_count > 0:
            tips.append({
                'type': 'success',
                'message': f"🎯 {high_profit_count} 只持仓盈利>10%，建议分批止盈锁定利润"
            })
        
        # 亏损警告
        if loss_count > 0:
            tips.append({
                'type': 'error',
                'message': f"⚠️  {loss_count} 只持仓亏损，注意及时止损，避免亏损扩大"
            })
        
        return tips
    
    def render_tips(self, stage: str, data: Dict[str, Any]):
        """渲染智能提示"""
        tips = self.generate_tips(stage, data)
        
        if not tips:
            return
        
        st.markdown("### 💡 智能提示")
        
        for tip in tips:
            tip_type = tip['type']
            message = tip['message']
            
            if tip_type == 'success':
                st.success(message)
            elif tip_type == 'info':
                st.info(message)
            elif tip_type == 'warning':
                st.warning(message)
            elif tip_type == 'error':
                st.error(message)


class ActionButtons:
    """一键操作按钮组"""
    
    def __init__(self, key_prefix: str = "action"):
        """初始化操作按钮组"""
        self.key_prefix = key_prefix
    
    def render_candidate_pool_actions(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        渲染候选池操作按钮
        
        Args:
            data: 候选池数据
            
        Returns:
            操作结果字典 {'saved': bool, 'exported': bool, 'reminded': bool}
        """
        if data.empty:
            st.info("📭 候选池为空，无法执行操作")
            return {'saved': False, 'exported': False, 'reminded': False}
        
        st.markdown("#### 🔧 快捷操作")
        
        col1, col2, col3, col4 = st.columns(4)
        
        results = {}
        
        with col1:
            if st.button(
                "💾 保存候选池",
                key=f"{self.key_prefix}_save",
                use_container_width=True,
                help="保存当前候选池到本地"
            ):
                results['saved'] = self._save_candidate_pool(data)
            else:
                results['saved'] = False
        
        with col2:
            if st.button(
                "📄 导出报告",
                key=f"{self.key_prefix}_export",
                use_container_width=True,
                help="导出Excel报告"
            ):
                results['exported'] = self._export_report(data)
            else:
                results['exported'] = False
        
        with col3:
            if st.button(
                "🔔 设置提醒",
                key=f"{self.key_prefix}_remind",
                use_container_width=True,
                help="设置竞价开盘提醒"
            ):
                results['reminded'] = self._set_reminder()
            else:
                results['reminded'] = False
        
        with col4:
            if st.button(
                "🔃 重新筛选",
                key=f"{self.key_prefix}_reset",
                use_container_width=True,
                help="清空筛选条件重新开始"
            ):
                st.rerun()
        
        return results
    
    def render_trading_actions(self, selected_stocks: List[str]) -> Dict[str, bool]:
        """
        渲染交易操作按钮
        
        Args:
            selected_stocks: 选中的股票列表
            
        Returns:
            操作结果字典
        """
        st.markdown("#### 💰 交易操作")
        
        if not selected_stocks:
            st.info("📭 请先选择股票")
            return {}
        
        col1, col2, col3 = st.columns(3)
        
        results = {}
        
        with col1:
            if st.button(
                "💵 模拟买入",
                key=f"{self.key_prefix}_buy",
                use_container_width=True,
                help="模拟买入选中股票"
            ):
                results['bought'] = self._simulate_buy(selected_stocks)
            else:
                results['bought'] = False
        
        with col2:
            if st.button(
                "💸 模拟卖出",
                key=f"{self.key_prefix}_sell",
                use_container_width=True,
                help="模拟卖出选中股票"
            ):
                results['sold'] = self._simulate_sell(selected_stocks)
            else:
                results['sold'] = False
        
        with col3:
            if st.button(
                "📊 查看详情",
                key=f"{self.key_prefix}_detail",
                use_container_width=True,
                help="查看股票详细信息"
            ):
                results['viewed'] = True
            else:
                results['viewed'] = False
        
        return results
    
    def _save_candidate_pool(self, data: pd.DataFrame) -> bool:
        """保存候选池"""
        try:
            output_dir = Path("output/candidate_pools")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"candidate_pool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = output_dir / filename
            
            # 转换为JSON
            data_dict = data.to_dict(orient='records')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(data),
                    'data': data_dict
                }, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 候选池已保存到 {filepath}")
            return True
        except Exception as e:
            st.error(f"❌ 保存失败: {e}")
            return False
    
    def _export_report(self, data: pd.DataFrame) -> bool:
        """导出Excel报告"""
        try:
            # 生成CSV下载
            csv = data.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 下载Excel报告",
                data=csv,
                file_name=f"stock_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"{self.key_prefix}_download"
            )
            
            return True
        except Exception as e:
            st.error(f"❌ 导出失败: {e}")
            return False
    
    def _set_reminder(self) -> bool:
        """设置提醒"""
        st.info("🔔 提醒功能开发中，将在竞价开盘前5分钟通知您")
        return True
    
    def _simulate_buy(self, stocks: List[str]) -> bool:
        """模拟买入"""
        st.success(f"✅ 模拟买入 {len(stocks)} 只股票: {', '.join(stocks[:3])}{'...' if len(stocks) > 3 else ''}")
        return True
    
    def _simulate_sell(self, stocks: List[str]) -> bool:
        """模拟卖出"""
        st.success(f"✅ 模拟卖出 {len(stocks)} 只股票: {', '.join(stocks[:3])}{'...' if len(stocks) > 3 else ''}")
        return True


class RiskLevelIndicator:
    """风险等级指示器"""
    
    @staticmethod
    def get_risk_level(profit_rate: float) -> Dict[str, str]:
        """
        根据盈亏率返回风险等级
        
        Args:
            profit_rate: 盈亏率 (%)
            
        Returns:
            {'level': 'high/medium/low', 'color': '...', 'emoji': '...', 'suggestion': '...'}
        """
        if profit_rate >= 10:
            return {
                'level': 'low',
                'color': 'green',
                'emoji': '🟢',
                'suggestion': '建议持有或分批止盈'
            }
        elif profit_rate >= 0:
            return {
                'level': 'medium',
                'color': 'yellow',
                'emoji': '🟡',
                'suggestion': '建议观望，关注走势'
            }
        elif profit_rate >= -5:
            return {
                'level': 'medium',
                'color': 'orange',
                'emoji': '🟠',
                'suggestion': '建议谨慎，考虑止损'
            }
        else:
            return {
                'level': 'high',
                'color': 'red',
                'emoji': '🔴',
                'suggestion': '建议立即止损'
            }
    
    @staticmethod
    def render_risk_badge(profit_rate: float):
        """渲染风险徽章"""
        risk = RiskLevelIndicator.get_risk_level(profit_rate)
        
        st.markdown(f"""
        <div style="
            display: inline-block;
            padding: 5px 10px;
            background-color: {risk['color']};
            color: white;
            border-radius: 5px;
            font-weight: bold;
        ">
            {risk['emoji']} {profit_rate:+.2f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(risk['suggestion'])


# 测试代码
if __name__ == "__main__":
    st.set_page_config(page_title="智能提示与操作测试", layout="wide")
    
    st.title("🤖 智能提示系统 & 一键操作测试")
    
    # 测试智能提示系统
    st.markdown("## 💡 智能提示系统测试")
    
    tip_system = SmartTipSystem()
    
    # 模拟不同阶段的数据
    test_stages = [
        ("T日选股", {
            'limitup_count': 85,
            'candidate_count': 12,
            'avg_quality_score': 75
        }),
        ("T+1竞价监控", {
            'strong_count': 5,
            'weak_count': 2,
            'avg_strength': 6.5,
            'monitor_count': 10
        }),
        ("T+2卖出决策", {
            'profit_count': 6,
            'loss_count': 2,
            'high_profit_count': 3
        })
    ]
    
    for stage, data in test_stages:
        st.markdown(f"### 阶段: {stage}")
        tip_system.render_tips(stage, data)
        st.markdown("---")
    
    # 测试操作按钮
    st.markdown("## 🔧 一键操作按钮测试")
    
    action_buttons = ActionButtons(key_prefix="test")
    
    # 模拟候选池数据
    test_data = pd.DataFrame({
        'symbol': ['000001', '000002', '000003'],
        'name': ['平安银行', '万科A', '国农科技'],
        'quality_score': [85, 78, 92]
    })
    
    action_buttons.render_candidate_pool_actions(test_data)
    
    st.markdown("---")
    
    # 测试交易操作
    test_stocks = ['000001', '000002']
    action_buttons.render_trading_actions(test_stocks)
    
    st.markdown("---")
    
    # 测试风险指示器
    st.markdown("## 🎯 风险等级指示器测试")
    
    test_profits = [15.5, 5.2, -3.1, -8.5]
    
    for profit in test_profits:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("盈亏率", f"{profit:+.2f}%")
        with col2:
            RiskLevelIndicator.render_risk_badge(profit)
