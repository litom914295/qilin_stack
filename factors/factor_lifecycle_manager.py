"""
因子生命周期管理器

根据 docs/IMPROVEMENT_ROADMAP.md 阶段一任务 1.2
目标：自动管理因子的生命周期状态和权重调整

核心功能：
1. 因子状态管理（活跃/观察/休眠/淘汰）
2. 自动降权机制（IC衰减时降低权重）
3. 自动淘汰机制（IC过低时送入冷宫）
4. 因子复活机制（休眠因子表现恢复时重新激活）
5. 状态转换规则引擎

作者：Qilin Quant Team
创建：2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FactorStatus(Enum):
    """因子状态枚举"""
    ACTIVE = "活跃"           # 正常使用，权重100%
    WATCHING = "观察"         # 表现下降，权重75%
    WARNING = "警告"          # 显著衰减，权重50%
    SLEEPING = "休眠"         # 暂停使用，权重0%
    ELIMINATED = "淘汰"       # 永久移除


class FactorLifecycleManager:
    """因子生命周期管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化因子生命周期管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or str(project_root / 'config' / 'factor_lifecycle_config.json')
        
        # 默认转换规则
        self.transition_rules = {
            'active_to_watching': {
                'ic_drop_threshold': 0.8,      # IC下降至历史均值的80%
                'win_rate_threshold': 0.52,    # 胜率低于52%
                'consecutive_bad_days': 20      # 连续20天表现不佳
            },
            'watching_to_warning': {
                'ic_drop_threshold': 0.5,       # IC下降至历史均值的50%
                'win_rate_threshold': 0.48,     # 胜率低于48%
                'consecutive_bad_days': 30
            },
            'warning_to_sleeping': {
                'ic_absolute_threshold': 0.01,  # IC绝对值低于0.01
                'consecutive_bad_days': 40,
                'ir_threshold': 0.3              # IR低于0.3
            },
            'sleeping_to_eliminated': {
                'sleep_duration_days': 120,     # 休眠超过120天
                'recovery_fail_count': 3         # 尝试复活3次均失败
            },
            'revival': {  # 复活条件
                'ic_recovery_threshold': 0.03,  # IC恢复至0.03以上
                'consecutive_good_days': 20,    # 连续20天表现良好
                'win_rate_threshold': 0.55      # 胜率高于55%
            }
        }
        
        # 权重配置
        self.weight_map = {
            FactorStatus.ACTIVE: 1.0,
            FactorStatus.WATCHING: 0.75,
            FactorStatus.WARNING: 0.5,
            FactorStatus.SLEEPING: 0.0,
            FactorStatus.ELIMINATED: 0.0
        }
        
        # 因子状态记录
        self.factor_states = {}  # {因子名: 状态信息}
        
        # 加载配置
        self._load_config()
        
        print("🔄 因子生命周期管理器初始化完成")
    
    def update_factor_status(self, 
                            factor_name: str, 
                            health_metrics: Dict,
                            force_update: bool = False) -> Dict:
        """
        更新因子状态
        
        Args:
            factor_name: 因子名称
            health_metrics: 健康度指标（来自FactorDecayMonitor）
            force_update: 是否强制更新
        
        Returns:
            Dict: 更新后的状态信息
        """
        # 获取当前状态
        current_state = self.factor_states.get(factor_name, {
            'status': FactorStatus.ACTIVE,
            'weight': 1.0,
            'history_ic_mean': health_metrics.get('ic_mean', 0.03),
            'bad_days_count': 0,
            'good_days_count': 0,
            'sleep_start_date': None,
            'revival_attempts': 0,
            'last_update': datetime.now(),
            'status_history': []
        })
        
        old_status = current_state['status']
        new_status = old_status
        
        # 根据当前状态和指标决定新状态
        ic_mean = health_metrics.get('ic_mean', 0)
        ic_recent = health_metrics.get('ic_recent', 0)
        win_rate = health_metrics.get('ic_win_rate', 0.5)
        ir = health_metrics.get('ir', 0)
        trend = health_metrics.get('ic_trend', 'stable')
        
        # 状态转换逻辑
        if old_status == FactorStatus.ACTIVE:
            new_status = self._check_active_to_watching(
                ic_mean, ic_recent, win_rate, 
                current_state['history_ic_mean'], 
                current_state
            )
        
        elif old_status == FactorStatus.WATCHING:
            # 可能升级回ACTIVE或降级到WARNING
            if self._check_revival_conditions(ic_mean, win_rate, trend, current_state):
                new_status = FactorStatus.ACTIVE
                current_state['good_days_count'] = 0
            else:
                new_status = self._check_watching_to_warning(
                    ic_mean, win_rate, 
                    current_state['history_ic_mean'], 
                    current_state
                )
        
        elif old_status == FactorStatus.WARNING:
            # 可能恢复到WATCHING或降级到SLEEPING
            if self._check_revival_conditions(ic_mean, win_rate, trend, current_state):
                new_status = FactorStatus.WATCHING
                current_state['good_days_count'] = 0
            else:
                new_status = self._check_warning_to_sleeping(
                    ic_mean, ir, current_state
                )
        
        elif old_status == FactorStatus.SLEEPING:
            # 检查是否可以复活或应该淘汰
            if self._check_revival_conditions(ic_mean, win_rate, trend, current_state, strict=True):
                new_status = FactorStatus.WATCHING
                current_state['revival_attempts'] += 1
                current_state['sleep_start_date'] = None
            else:
                new_status = self._check_sleeping_to_eliminated(current_state)
        
        elif old_status == FactorStatus.ELIMINATED:
            # 淘汰后不再改变状态
            pass
        
        # 更新状态记录
        if new_status != old_status or force_update:
            current_state['status'] = new_status
            current_state['weight'] = self.weight_map[new_status]
            current_state['last_update'] = datetime.now()
            
            # 记录状态转换历史
            current_state['status_history'].append({
                'date': datetime.now(),
                'from_status': old_status.value if isinstance(old_status, FactorStatus) else old_status,
                'to_status': new_status.value,
                'reason': self._get_transition_reason(old_status, new_status, health_metrics)
            })
            
            # 如果进入休眠，记录开始时间
            if new_status == FactorStatus.SLEEPING and old_status != FactorStatus.SLEEPING:
                current_state['sleep_start_date'] = datetime.now()
            
            print(f"  ⚙️ 因子 {factor_name} 状态变更: {old_status.value if isinstance(old_status, FactorStatus) else old_status} -> {new_status.value}")
        
        # 保存状态
        self.factor_states[factor_name] = current_state
        
        return current_state
    
    def batch_update_factors(self, health_df: pd.DataFrame) -> pd.DataFrame:
        """
        批量更新多个因子状态
        
        Args:
            health_df: 健康度DataFrame（来自FactorDecayMonitor）
        
        Returns:
            pd.DataFrame: 包含状态和权重的DataFrame
        """
        print(f"\n批量更新 {len(health_df)} 个因子状态...")
        
        results = []
        
        for _, row in health_df.iterrows():
            factor_name = row['factor_name']
            
            # 转换为字典
            health_metrics = row.to_dict()
            
            # 更新状态
            state_info = self.update_factor_status(factor_name, health_metrics)
            
            # 合并结果
            result = {
                'factor_name': factor_name,
                'status': state_info['status'].value,
                'weight': state_info['weight'],
                'ic_mean': health_metrics.get('ic_mean', 0),
                'win_rate': health_metrics.get('ic_win_rate', 0),
                'ir': health_metrics.get('ir', 0),
                'last_update': state_info['last_update']
            }
            results.append(result)
        
        df_results = pd.DataFrame(results)
        
        return df_results
    
    def get_active_factors(self) -> List[str]:
        """获取所有活跃因子列表"""
        active_factors = [
            name for name, state in self.factor_states.items()
            if state['status'] in [FactorStatus.ACTIVE, FactorStatus.WATCHING, FactorStatus.WARNING]
        ]
        return active_factors
    
    def get_factor_weights(self) -> Dict[str, float]:
        """获取所有因子的当前权重"""
        weights = {
            name: state['weight']
            for name, state in self.factor_states.items()
        }
        return weights
    
    def generate_lifecycle_report(self, output_path: str = None) -> str:
        """
        生成生命周期管理报告
        
        Args:
            output_path: 输出路径
        
        Returns:
            str: 报告内容
        """
        print("\n生成因子生命周期报告...")
        
        report_lines = []
        report_lines.append("# 因子生命周期管理报告\n\n")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**管理因子数**: {len(self.factor_states)}\n\n")
        
        # 统计各状态因子数
        status_counts = {}
        for state_info in self.factor_states.values():
            status = state_info['status'].value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        report_lines.append("## 📊 状态分布\n\n")
        for status, count in status_counts.items():
            report_lines.append(f"- **{status}**: {count}个因子\n")
        report_lines.append("\n")
        
        # 按状态分组展示
        for status_enum in [FactorStatus.ACTIVE, FactorStatus.WATCHING, 
                           FactorStatus.WARNING, FactorStatus.SLEEPING, 
                           FactorStatus.ELIMINATED]:
            factors_in_status = [
                (name, state) for name, state in self.factor_states.items()
                if state['status'] == status_enum
            ]
            
            if factors_in_status:
                report_lines.append(f"## {status_enum.value}因子\n\n")
                report_lines.append("| 因子名称 | 权重 | 历史IC | 上次更新 | 状态持续天数 |\n")
                report_lines.append("|---------|-----|--------|---------|-------------|\n")
                
                for name, state in factors_in_status:
                    days_in_status = (datetime.now() - state['last_update']).days
                    report_lines.append(
                        f"| {name} "
                        f"| {state['weight']:.2f} "
                        f"| {state['history_ic_mean']:.4f} "
                        f"| {state['last_update'].strftime('%Y-%m-%d')} "
                        f"| {days_in_status}天 |\n"
                    )
                
                report_lines.append("\n")
        
        # 最近状态转换记录
        report_lines.append("## 📝 最近状态转换记录\n\n")
        
        all_transitions = []
        for name, state in self.factor_states.items():
            for trans in state['status_history'][-3:]:  # 最近3次转换
                all_transitions.append({
                    'factor_name': name,
                    **trans
                })
        
        # 按时间排序
        all_transitions.sort(key=lambda x: x['date'], reverse=True)
        
        if all_transitions:
            report_lines.append("| 日期 | 因子 | 状态变化 | 原因 |\n")
            report_lines.append("|------|------|---------|------|\n")
            
            for trans in all_transitions[:10]:  # 显示最近10条
                report_lines.append(
                    f"| {trans['date'].strftime('%Y-%m-%d')} "
                    f"| {trans['factor_name']} "
                    f"| {trans['from_status']} → {trans['to_status']} "
                    f"| {trans['reason']} |\n"
                )
        else:
            report_lines.append("暂无状态转换记录\n")
        
        report_content = "".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"✅ 报告已保存至: {output_path}")
        
        return report_content
    
    def save_state(self, filepath: str = None):
        """保存因子状态到文件"""
        if filepath is None:
            filepath = project_root / 'data' / 'factor_states.json'
        
        # 序列化状态（枚举转字符串）
        serializable_states = {}
        for name, state in self.factor_states.items():
            serializable_states[name] = {
                'status': state['status'].value,
                'weight': state['weight'],
                'history_ic_mean': state['history_ic_mean'],
                'bad_days_count': state['bad_days_count'],
                'good_days_count': state['good_days_count'],
                'sleep_start_date': state['sleep_start_date'].isoformat() if state['sleep_start_date'] else None,
                'revival_attempts': state['revival_attempts'],
                'last_update': state['last_update'].isoformat(),
                'status_history': [
                    {**h, 'date': h['date'].isoformat()} for h in state['status_history']
                ]
            }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_states, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 因子状态已保存至: {filepath}")
    
    def load_state(self, filepath: str = None):
        """从文件加载因子状态"""
        if filepath is None:
            filepath = project_root / 'data' / 'factor_states.json'
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"⚠️ 状态文件不存在: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            serializable_states = json.load(f)
        
        # 反序列化
        for name, state in serializable_states.items():
            self.factor_states[name] = {
                'status': FactorStatus(state['status']),
                'weight': state['weight'],
                'history_ic_mean': state['history_ic_mean'],
                'bad_days_count': state['bad_days_count'],
                'good_days_count': state['good_days_count'],
                'sleep_start_date': datetime.fromisoformat(state['sleep_start_date']) if state['sleep_start_date'] else None,
                'revival_attempts': state['revival_attempts'],
                'last_update': datetime.fromisoformat(state['last_update']),
                'status_history': [
                    {**h, 'date': datetime.fromisoformat(h['date'])} for h in state['status_history']
                ]
            }
        
        print(f"✅ 已加载 {len(self.factor_states)} 个因子状态")
    
    # ==================== 内部方法 ====================
    
    def _check_active_to_watching(self, ic_mean, ic_recent, win_rate, history_ic_mean, state) -> FactorStatus:
        """检查是否从活跃转为观察"""
        rules = self.transition_rules['active_to_watching']
        
        # IC显著下降
        if ic_mean < history_ic_mean * rules['ic_drop_threshold']:
            state['bad_days_count'] += 1
        else:
            state['bad_days_count'] = 0
        
        # 胜率过低
        if win_rate < rules['win_rate_threshold']:
            state['bad_days_count'] += 1
        
        # 连续表现不佳
        if state['bad_days_count'] >= rules['consecutive_bad_days']:
            return FactorStatus.WATCHING
        
        return FactorStatus.ACTIVE
    
    def _check_watching_to_warning(self, ic_mean, win_rate, history_ic_mean, state) -> FactorStatus:
        """检查是否从观察转为警告"""
        rules = self.transition_rules['watching_to_warning']
        
        if ic_mean < history_ic_mean * rules['ic_drop_threshold'] or win_rate < rules['win_rate_threshold']:
            state['bad_days_count'] += 1
        else:
            state['bad_days_count'] = max(0, state['bad_days_count'] - 1)
        
        if state['bad_days_count'] >= rules['consecutive_bad_days']:
            return FactorStatus.WARNING
        
        return FactorStatus.WATCHING
    
    def _check_warning_to_sleeping(self, ic_mean, ir, state) -> FactorStatus:
        """检查是否从警告转为休眠"""
        rules = self.transition_rules['warning_to_sleeping']
        
        if ic_mean < rules['ic_absolute_threshold'] or ir < rules['ir_threshold']:
            state['bad_days_count'] += 1
        else:
            state['bad_days_count'] = max(0, state['bad_days_count'] - 1)
        
        if state['bad_days_count'] >= rules['consecutive_bad_days']:
            return FactorStatus.SLEEPING
        
        return FactorStatus.WARNING
    
    def _check_sleeping_to_eliminated(self, state) -> FactorStatus:
        """检查是否从休眠转为淘汰"""
        rules = self.transition_rules['sleeping_to_eliminated']
        
        if state['sleep_start_date']:
            sleep_days = (datetime.now() - state['sleep_start_date']).days
            
            if sleep_days > rules['sleep_duration_days']:
                return FactorStatus.ELIMINATED
            
            if state['revival_attempts'] >= rules['recovery_fail_count']:
                return FactorStatus.ELIMINATED
        
        return FactorStatus.SLEEPING
    
    def _check_revival_conditions(self, ic_mean, win_rate, trend, state, strict=False) -> bool:
        """检查是否满足复活条件"""
        rules = self.transition_rules['revival']
        
        if ic_mean >= rules['ic_recovery_threshold'] and win_rate >= rules['win_rate_threshold']:
            state['good_days_count'] += 1
        else:
            state['good_days_count'] = 0
        
        threshold = rules['consecutive_good_days'] if not strict else rules['consecutive_good_days'] * 1.5
        
        if state['good_days_count'] >= threshold:
            return True
        
        return False
    
    def _get_transition_reason(self, old_status, new_status, metrics) -> str:
        """获取状态转换原因"""
        if isinstance(old_status, FactorStatus):
            old = old_status.value
        else:
            old = old_status
        
        new = new_status.value
        
        ic_mean = metrics.get('ic_mean', 0)
        win_rate = metrics.get('ic_win_rate', 0)
        
        reasons = {
            ('活跃', '观察'): f'IC下降至{ic_mean:.4f}, 胜率{win_rate:.2%}',
            ('观察', '警告'): f'持续表现不佳, IC={ic_mean:.4f}',
            ('警告', '休眠'): f'IC过低({ic_mean:.4f})或IR不足',
            ('休眠', '淘汰'): '休眠期过长或复活失败次数过多',
            ('观察', '活跃'): f'表现恢复, IC={ic_mean:.4f}',
            ('警告', '观察'): f'表现好转, 胜率{win_rate:.2%}',
            ('休眠', '观察'): f'复活成功, IC恢复至{ic_mean:.4f}'
        }
        
        return reasons.get((old, new), '状态转换')
    
    def _load_config(self):
        """加载配置文件"""
        config_path = Path(self.config_path)
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                if 'transition_rules' in config:
                    self.transition_rules.update(config['transition_rules'])
                
                if 'weight_map' in config:
                    # 转换为FactorStatus枚举
                    for status_str, weight in config['weight_map'].items():
                        status = FactorStatus(status_str)
                        self.weight_map[status] = weight
            
            print(f"  已加载配置: {config_path}")


def main():
    """主函数 - 示例用法"""
    from monitoring.factor_decay_monitor import FactorDecayMonitor
    
    # 初始化管理器
    manager = FactorLifecycleManager()
    monitor = FactorDecayMonitor()
    
    # 模拟健康度数据
    health_data = {
        'factor_momentum': {'ic_mean': 0.04, 'ic_recent': 0.035, 'ic_win_rate': 0.58, 'ir': 1.2, 'ic_trend': 'stable'},
        'factor_value': {'ic_mean': 0.015, 'ic_recent': 0.012, 'ic_win_rate': 0.48, 'ir': 0.6, 'ic_trend': 'declining'},
        'factor_quality': {'ic_mean': 0.008, 'ic_recent': 0.006, 'ic_win_rate': 0.45, 'ir': 0.3, 'ic_trend': 'declining'},
    }
    
    print("\n" + "="*70)
    print("🔄 因子生命周期管理演示")
    print("="*70)
    
    # 更新因子状态
    for factor_name, metrics in health_data.items():
        metrics['factor_name'] = factor_name
        state = manager.update_factor_status(factor_name, metrics)
        print(f"\n因子: {factor_name}")
        print(f"  状态: {state['status'].value}")
        print(f"  权重: {state['weight']}")
    
    # 获取活跃因子
    active_factors = manager.get_active_factors()
    print(f"\n活跃因子列表: {active_factors}")
    
    # 获取因子权重
    weights = manager.get_factor_weights()
    print(f"\n因子权重: {weights}")
    
    # 生成报告
    report_path = project_root / 'reports' / 'factor_lifecycle_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manager.generate_lifecycle_report(str(report_path))
    
    # 保存状态
    manager.save_state()
    
    print("\n✅ 生命周期管理演示完成！")


if __name__ == '__main__':
    main()
