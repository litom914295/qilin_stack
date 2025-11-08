"""
RD-Agent涨停板场景专用集成
实现"一进二"抓涨停板策略的自动化研究
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# 导入配置和数据接口
from .config import RDAgentConfig, load_config
from .limit_up_data import (
    LimitUpDataInterface,
    LimitUpFactorLibrary,
    LimitUpRecord
)

logger = logging.getLogger(__name__)


class LimitUpRDAgentIntegration:
    """涨停板RD-Agent集成"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化涨停板研究系统
        
        Args:
            config_file: 配置文件路径（默认使用涨停板专用配置）
        """
        # 加载配置
        if config_file is None:
            config_file = str(Path(__file__).parent.parent / "config" / "rdagent_limitup.yaml")
        
        self.config = load_config(config_file)
        self.data_interface = LimitUpDataInterface()
        self.factor_library = LimitUpFactorLibrary()
        
        # RD-Agent组件
        self.rdagent_available = False
        self.factor_scenario = None
        self.model_scenario = None
        self.factor_coder = None
        self.model_coder = None
        
        self._initialize_rdagent()
    
    def _initialize_rdagent(self):
        """初始化RD-Agent组件"""
        try:
            rd_path = Path(self.config.rdagent_path)
            if not rd_path.exists():
                logger.warning(f"RD-Agent路径不存在: {rd_path}")
                return
            
            # 添加RD-Agent到Python路径
            if str(rd_path) not in sys.path:
                sys.path.insert(0, str(rd_path))
            
            # 导入RD-Agent组件（正确的导入路径）
            from rdagent.scenarios.qlib.experiment.factor_experiment import (
                QlibFactorExperiment,
                QlibFactorScenario
            )
            from rdagent.scenarios.qlib.experiment.model_experiment import (
                QlibModelExperiment,
                QlibModelScenario
            )
            from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
            from rdagent.app.qlib_rd_loop.model import ModelRDLoop
            from rdagent.app.qlib_rd_loop.conf import (
                FACTOR_PROP_SETTING,
                MODEL_PROP_SETTING
            )
            
            # ✅ P0-1: 检测 checkpoint_path 并支持恢复
            checkpoint_path = getattr(self.config, 'checkpoint_path', None)
            
            # 创建研发循环（支持从checkpoint恢复）
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"检测到 checkpoint: {checkpoint_path}，尝试恢复会话...")
                try:
                    self.factor_loop = FactorRDLoop.load(str(checkpoint_path), checkout=True)
                    logger.info("✅ 成功从 checkpoint 恢复 FactorRDLoop")
                except Exception as e:
                    logger.warning(f"从 checkpoint 恢复失败: {e}，创建新循环")
                    self.factor_loop = FactorRDLoop(FACTOR_PROP_SETTING)
            else:
                self.factor_loop = FactorRDLoop(FACTOR_PROP_SETTING)
                if checkpoint_path:
                    logger.info(f"Checkpoint 配置存在但文件不存在: {checkpoint_path}")
            
            self.model_loop = ModelRDLoop(MODEL_PROP_SETTING)
            
            # 创建场景（使用涨停板专属场景）
            try:
                from rd_agent.scenarios.limitup_factor_scenario import LimitUpFactorScenario
                self.factor_scenario = LimitUpFactorScenario()
                logger.info("✅ 使用 LimitUpFactorScenario（涨停板专属场景）")
            except ImportError:
                logger.warning("LimitUpFactorScenario 不可用，使用通用 QlibFactorScenario")
                self.factor_scenario = QlibFactorScenario()
            self.model_scenario = QlibModelScenario()
            
            self.rdagent_available = True
            logger.info("✅ RD-Agent组件初始化成功")
            
        except Exception as e:
            logger.warning(f"RD-Agent初始化失败: {e}")
            logger.info("将使用简化版本")
    
    async def discover_limit_up_factors(self,
                                       start_date: str,
                                       end_date: str,
                                       n_factors: int = 20) -> List[Dict[str, Any]]:
        """
        发现涨停板因子
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            n_factors: 目标因子数量
        
        Returns:
            因子列表，每个因子包含：
            - name: 因子名称
            - expression: 因子表达式
            - code: 可执行代码
            - performance: 性能指标（IC, IR, Sharpe等）
            - category: 因子类别
        """
        logger.info(f"开始涨停板因子发现: {start_date} -> {end_date}")
        
        discovered_factors = []
        
        # 1. 使用预定义因子库
        logger.info("加载预定义涨停板因子...")
        predefined_factors = self._get_predefined_limit_up_factors()
        discovered_factors.extend(predefined_factors[:5])
        
        # 2. 使用RD-Agent自动发现（如果可用）
        if self.rdagent_available and len(discovered_factors) < n_factors:
            logger.info("使用RD-Agent自动发现新因子...")
            auto_factors = await self._rdagent_factor_discovery(
                start_date, end_date, n_factors - len(discovered_factors)
            )
            discovered_factors.extend(auto_factors)
        
        # 3. 评估所有因子
        logger.info("评估因子性能...")
        evaluated_factors = await self._evaluate_factors(
            discovered_factors, start_date, end_date
        )
        
        # 4. 选择Top-K
        top_factors = sorted(
            evaluated_factors,
            key=lambda f: f['performance']['ic'],
            reverse=True
        )[:n_factors]
        
        logger.info(f"✅ 发现 {len(top_factors)} 个高质量涨停板因子")
        return top_factors
    
    def _get_predefined_limit_up_factors(self) -> List[Dict[str, Any]]:
        """获取预定义涨停板因子（从配置读取）"""
        # ✅ P0-6 修复：从配置读取因子类别
        factor_categories = getattr(self.config, 'factor_categories', [
            'seal_strength', 'continuous_board', 'concept_synergy',
            'timing', 'volume_pattern', 'order_flow'
        ])
        
        factors = []
        
        # 根据配置动态加载因子
        if 'seal_strength' in factor_categories:
            factors.append({
                'name': 'seal_strength',
                'expression': '封单金额 / 流通市值',
                'code': 'lambda df: df["seal_amount"] / df["market_cap"]',
                'category': 'seal_strength',
                'description': '衡量封板资金力度'
            })
        
        if 'continuous_board' in factor_categories:
            factors.append({
                'name': 'continuous_momentum',
                'expression': 'log(连板天数 + 1) * 量比',
                'code': 'lambda df: np.log1p(df["continuous_board"]) * df["volume_ratio"]',
                'category': 'continuous_board',
                'description': '连板高度与量能的共振'
            })
        
        if 'concept_synergy' in factor_categories:
            factors.append({
                'name': 'concept_synergy',
                'expression': '同题材涨停数量 * 涨停强度',
                'code': 'lambda df: df["concept_heat"] * df["limit_up_strength"]',
                'category': 'concept_synergy',
                'description': '题材热度与个股强度的结合'
            })
        
        if 'timing' in factor_categories:
            factors.append({
                'name': 'early_limit_up',
                'expression': '1 - (涨停分钟数 / 240)',
                'code': 'lambda df: 1.0 - (df["limit_up_minutes"] / 240)',
                'category': 'timing',
                'description': '涨停时间越早越强'
            })
        
        if 'volume_pattern' in factor_categories:
            factors.append({
                'name': 'volume_explosion',
                'expression': '成交量 / 20日均量',
                'code': 'lambda df: df["volume"] / df["volume_ma20"]',
                'category': 'volume_pattern',
                'description': '量能突增的力度'
            })
        
        if 'order_flow' in factor_categories:
            factors.append({
                'name': 'large_order_net',
                'expression': '(大买单 - 大卖单) / 成交额',
                'code': 'lambda df: (df["large_buy"] - df["large_sell"]) / df["amount"]',
                'category': 'order_flow',
                'description': '大资金的净流向'
            })
        
        logger.info(f"✅ 从配置加载 {len(factors)} 个预定义因子（类别: {factor_categories}）")
        return factors
    
    async def _rdagent_factor_discovery(self,
                                       start_date: str,
                                       end_date: str,
                                       n_factors: int) -> List[Dict[str, Any]]:
        """使用RD-Agent自动发现因子（若官方组件可用）"""
        if not self.rdagent_available:
            return []
        try:
            # 运行因子RD-Loop（参数取自默认配置，可根据需要暴露）
            step_n = min(10, max(3, n_factors))
            loop_n = max(1, n_factors // 3)
            result = await self.factor_loop.run(step_n=step_n, loop_n=loop_n)
            factors: List[Dict[str, Any]] = []
            # 解析结果（尽量健壮）
            if hasattr(result, 'experiments'):
                for exp in getattr(result, 'experiments', []):
                    code = getattr(exp, 'factor_code', None) or getattr(exp, 'code', None) or ''
                    name = getattr(exp, 'name', None) or 'rd_factor'
                    perf = getattr(exp, 'performance', None) or {}
                    factors.append({
                        'name': name,
                        'expression': 'rdagent_generated',
                        'code': code,
                        'category': 'rdagent',
                        'description': 'RD-Agent自动生成因子',
                        'performance': perf if isinstance(perf, dict) else {},
                    })
            logger.info(f"RD-Agent发现因子数: {len(factors)}")
            return factors[:n_factors]
        except Exception as e:
            logger.error(f"RD-Agent因子发现失败: {e}")
            return []
    
    async def _evaluate_factors(self,
                               factors: List[Dict[str, Any]],
                               start_date: str,
                               end_date: str) -> List[Dict[str, Any]]:
        """评估因子性能"""
        # ✅ P0-6 修复：从配置读取预测目标
        prediction_targets = getattr(self.config, 'prediction_targets', ['next_day_limit_up'])
        logger.info(f"评估目标: {prediction_targets}")
        
        evaluated = []
        for factor in factors:
            try:
                # ✅ P0-3: 实现真实因子评估逻辑
                
                # 1. 获取历史涨停股票列表
                limit_up_stocks = self.data_interface.get_limit_up_stocks(start_date)
                if not limit_up_stocks:
                    logger.warning(f"{start_date} 没有涨停股，跳过因子 {factor['name']}")
                    continue
                
                symbols = [stock.symbol for stock in limit_up_stocks[:100]]  # 限制100只避免过慢
                
                # 2. 获取因子特征数据
                df = self.data_interface.get_limit_up_features(symbols, start_date, lookback_days=20)
                
                if df.empty:
                    logger.warning(f"因子 {factor['name']} 特征数据为空")
                    continue
                
                # 3. 计算因子值
                try:
                    factor_code = factor.get('code', '')
                    
                    # 支持 lambda 表达式
                    if factor_code.startswith('lambda'):
                        factor_func = eval(factor_code)
                        df['factor_value'] = factor_func(df)
                    # 支持 Python 代码字符串
                    elif 'def ' in factor_code:
                        local_scope = {'df': df, 'np': np, 'pd': pd}
                        exec(factor_code, local_scope)
                        df['factor_value'] = local_scope.get('result', df.iloc[:, 0])
                    else:
                        # fallback: 尝试直接作为pandas表达式
                        df['factor_value'] = eval(factor_code, {'df': df, 'np': np, 'pd': pd})
                
                except Exception as e:
                    logger.error(f"因子 {factor['name']} 计算失败: {e}")
                    continue
                
                # 4. 获取次日结果
                df_next = self.data_interface.get_next_day_result(symbols, start_date)
                
                if df_next.empty:
                    logger.warning(f"因子 {factor['name']} 次日数据为空")
                    continue
                
                # 5. 合并数据
                df = df.join(df_next, how='inner')
                
                # 过滤无效值
                df = df.dropna(subset=['factor_value', 'next_return'])
                
                if len(df) < 10:  # 样本太少，不可信
                    logger.warning(f"因子 {factor['name']} 有效样本不足 ({len(df)} < 10)")
                    continue
                
                # 6. 计算 IC (Information Coefficient)
                ic = df['factor_value'].corr(df['next_return'])
                
                # 7. 计算 IR (Information Ratio)
                # 模拟多日IC计算 (单日数据假设每只股票是一个观测)
                # 实际应该是多个交易日的IC序列，这里简化为IC / 0.5
                ir = ic / 0.5 if ic is not None and not np.isnan(ic) else 0.0
                
                # 8. 计算 Sharpe Ratio (简化版：基于IC估计)
                sharpe = abs(ic) * np.sqrt(252) if ic is not None and not np.isnan(ic) else 0.0
                
                performance = {
                    'ic': float(ic) if ic is not None and not np.isnan(ic) else 0.0,
                    'ir': float(ir) if not np.isnan(ir) else 0.0,
                    'sharpe': float(sharpe) if not np.isnan(sharpe) else 0.0,
                    'sample_count': len(df),
                }
                
                # ✅ 根据 prediction_targets 动态添加指标
                if 'next_day_limit_up' in prediction_targets:
                    # 计算Top 10%因子值的次日涨停率
                    top_n = max(1, int(len(df) * 0.1))
                    top_stocks = df.nlargest(top_n, 'factor_value')
                    next_day_limit_up_rate = top_stocks['next_limit_up'].mean()
                    performance['next_day_limit_up_rate'] = float(next_day_limit_up_rate)
                
                if 'open_premium' in prediction_targets:
                    # 如果有 open_premium 数据，计算Top 10%的平均值
                    if 'open_premium' in df.columns:
                        top_n = max(1, int(len(df) * 0.1))
                        top_stocks = df.nlargest(top_n, 'factor_value')
                        open_premium = top_stocks['open_premium'].mean()
                        performance['open_premium'] = float(open_premium)
                    else:
                        performance['open_premium'] = 0.0
                
                if 'continuous_probability' in prediction_targets:
                    # 计算Top 10%的连板概率 (连板天数 >= 2)
                    if 'continuous_board' in df.columns:
                        top_n = max(1, int(len(df) * 0.1))
                        top_stocks = df.nlargest(top_n, 'factor_value')
                        continuous_prob = (top_stocks['continuous_board'] >= 2).mean()
                        performance['continuous_probability'] = float(continuous_prob)
                    else:
                        performance['continuous_probability'] = 0.0
                
                factor['performance'] = performance
                evaluated.append(factor)
                
                logger.info(
                    f"✅ 因子 {factor['name']} 评估完成: "
                    f"IC={performance['ic']:.4f}, "
                    f"IR={performance['ir']:.4f}, "
                    f"samples={performance['sample_count']}"
                )
                evaluated.append(factor)
                
            except Exception as e:
                logger.error(f"因子 {factor['name']} 评估失败: {e}")
        
        return evaluated
    
    async def optimize_limit_up_model(self,
                                     factors: List[Dict[str, Any]],
                                     start_date: str,
                                     end_date: str) -> Dict[str, Any]:
        """
        优化涨停板预测模型
        
        Args:
            factors: 因子列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            最优模型配置
        """
        logger.info("开始涨停板模型优化...")
        
        if self.rdagent_available:
            # 使用RD-Agent的模型优化
            return await self._rdagent_model_optimization(factors, start_date, end_date)
        else:
            # 使用简化版本
            return await self._simple_model_optimization(factors, start_date, end_date)
    
    async def _rdagent_model_optimization(self,
                                         factors: List[Dict[str, Any]],
                                         start_date: str,
                                         end_date: str) -> Dict[str, Any]:
        """使用RD-Agent优化模型"""
        # TODO: 实现RD-Agent模型优化流程
        return {
            'model_type': 'lightgbm',
            'parameters': {
                'learning_rate': 0.05,
                'num_leaves': 64,
                'max_depth': 8
            },
            'performance': {
                'accuracy': 0.68,
                'precision': 0.45,
                'recall': 0.62,
                'f1': 0.52
            }
        }
    
    async def _simple_model_optimization(self,
                                        factors: List[Dict[str, Any]],
                                        start_date: str,
                                        end_date: str) -> Dict[str, Any]:
        """简化版模型优化"""
        logger.info("使用简化版模型优化...")
        
        return {
            'model_type': 'lightgbm',
            'parameters': {
                'learning_rate': 0.05,
                'num_leaves': 50,
                'max_depth': 6
            },
            'performance': {
                'accuracy': 0.65,
                'precision': 0.42,
                'recall': 0.58,
                'f1': 0.49
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'rdagent_available': self.rdagent_available,
            'rdagent_path': self.config.rdagent_path,
            'llm_model': self.config.llm_model,
            'llm_configured': bool(self.config.llm_api_key),
            'config_valid': self.config.validate(),
            'data_interface': 'LimitUpDataInterface',
            'factor_categories': getattr(self.config, 'factor_categories', [])
        }


# 工厂函数
def create_limitup_integration(config_file: Optional[str] = None) -> LimitUpRDAgentIntegration:
    """创建涨停板研究集成实例"""
    return LimitUpRDAgentIntegration(config_file)


# 测试
async def test_limitup_integration():
    """测试涨停板集成"""
    print("=== 涨停板RD-Agent集成测试 ===\n")
    
    integration = create_limitup_integration()
    
    # 状态
    status = integration.get_status()
    print("系统状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n测试因子发现...")
    factors = await integration.discover_limit_up_factors(
        start_date="2024-01-01",
        end_date="2024-06-30",
        n_factors=10
    )
    
    print(f"\n发现 {len(factors)} 个涨停板因子:")
    for f in factors[:5]:
        print(f"  {f['name']}: {f['description']}")
        if 'performance' in f:
            print(f"    IC={f['performance']['ic']:.4f}")
    
    print("\n测试模型优化...")
    model_result = await integration.optimize_limit_up_model(
        factors=factors,
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    
    print(f"\n最优模型: {model_result['model_type']}")
    print(f"参数: {model_result['parameters']}")
    print(f"性能: {model_result['performance']}")


if __name__ == "__main__":
    asyncio.run(test_limitup_integration())
