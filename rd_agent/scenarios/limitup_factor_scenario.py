"""
LimitUp Factor Scenario - 涨停板因子场景

A-股"一进二"策略专属场景，基于官方 QlibFactorScenario 扩展，
增加涨停板特有的 Prompt 模板与数据字段说明。

Author: Qilin Stack Team
Created: 2025-01-07
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from rdagent.core.scenario import Scenario
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
    from rdagent.utils.agent.tpl import T
except ImportError as e:
    print(f"警告: 无法导入 RD-Agent 官方模块 ({e})，请确保 RDAGENT_PATH 正确")
    # 占位基类，避免导入失败导致整个模块无法加载
    class Scenario:
        pass
    class QlibFactorScenario(Scenario):
        pass


class LimitUpFactorScenario(QlibFactorScenario):
    """
    涨停板因子场景（A股"一进二"策略）
    
    核心特性：
    - 涨停板专属 Prompt（封单/连板/题材/时机/量能/资金）
    - A股特化数据字段说明
    - 次日涨停率/连板概率等预测目标
    
    继承官方 QlibFactorScenario，覆盖 Prompt 模板部分。
    """
    
    def __init__(self, **kwargs):
        """
        初始化涨停板因子场景
        
        Args:
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        # 加载涨停板专属 Prompt 模板
        self._load_limitup_prompts()
    
    def _load_limitup_prompts(self):
        """从 prompts_limitup.yaml 加载涨停板专属 Prompt"""
        try:
            # 获取当前文件所在目录
            scenarios_dir = Path(__file__).parent
            prompts_file = scenarios_dir / "prompts_limitup.yaml"
            
            if not prompts_file.exists():
                print(f"警告: prompts_limitup.yaml 不存在于 {prompts_file}，使用默认 Prompt")
                self._use_default_prompts()
                return
            
            # 使用官方模板引擎加载
            self._background = T(f"{prompts_file}:limitup_factor_background").r(
                runtime_environment=self.get_runtime_environment()
            )
            self._source_data = T(f"{prompts_file}:limitup_factor_source_data").r()
            self._interface = T(f"{prompts_file}:limitup_factor_interface").r()
            self._output_format = T(f"{prompts_file}:limitup_factor_output_format").r()
            self._simulator = T(f"{prompts_file}:limitup_factor_simulator").r()
            self._strategy = T(f"{prompts_file}:limitup_factor_strategy").r()
            self._experiment_setting = T(f"{prompts_file}:limitup_factor_experiment_setting").r()
            
            print(f"✅ 涨停板 Prompt 模板加载成功: {prompts_file}")
            
        except Exception as e:
            print(f"警告: 加载 prompts_limitup.yaml 失败 ({e})，使用默认 Prompt")
            self._use_default_prompts()
    
    def _use_default_prompts(self):
        """使用内置默认 Prompt（当 YAML 文件不存在时）"""
        self._background = """
涨停板（Limit Up）是 A股市场特有的交易机制，每日涨幅限制为 10%（ST 股 5%/20%，北交所 30%，科创板 20%）。
"一进二"策略：寻找今日涨停、明日有望继续涨停或高溢价开盘的股票。

关键因素：
1. **封单强度**：封单金额越大，资金承接力越强
2. **连板高度**：连板天数反映市场情绪延续性
3. **题材共振**：同题材涨停股数量体现板块热度
4. **涨停时机**：早盘涨停强度优于尾盘
5. **量能爆发**：成交量突增反映资金进场力度
6. **资金承接**：大单净流入体现主力态度

您需要生成能够预测以下目标的因子：
- next_day_limit_up: 次日是否涨停（二分类）
- next_day_high_return: 次日最高收益率
- open_premium: 开盘溢价百分比
- continuous_probability: 继续连板概率
        """
        
        self._source_data = """
可用数据字段（基于 Qlib + AKShare）：

**基础量价**（Qlib）：
- open, close, high, low, volume, amount, vwap
- 技术指标：ma5, ma10, ma20, rsi, macd, ...

**涨停板特有**（LimitUpDataInterface）：
- seal_amount: 封单金额（万元）
- seal_quality: 封板质量（封单/流通市值）
- continuous_board: 连板天数（0=首板，1=二板，...）
- limit_up_minutes: 涨停时刻（距开盘分钟数，0-240）
- concept_heat: 同题材涨停股数量
- large_buy / large_sell: 大单买入/卖出金额（万元）
- open_count: 开板次数（0=一字板，>=1=开板涨停）

**A股特化**：
- is_st: 是否 ST/*ST 股（高风险）
- list_days: 上市天数（<60 为次新股）
- industry: 行业分类（申万一级）
- concept_tags: 题材标签列表（如 ["新能源", "锂电池"]）
- turnover_rate: 换手率

**注意事项**：
- 涨停识别：close / prev_close - 1 >= 0.095（考虑浮点误差）
- 一字板排除：open == close == high == low 或 seal_amount == 0
- ST 股涨幅：5%（老ST）或 20%（*ST）
- 北交所/科创板：30%/20% 涨幅
        """
    
    @property
    def background(self) -> str:
        """返回场景背景说明"""
        return self._background
    
    def get_source_data_desc(self, task: Optional[Any] = None) -> str:
        """返回数据源说明"""
        return self._source_data
    
    def get_interface_description(self) -> str:
        """返回接口说明"""
        return getattr(self, '_interface', super().get_interface_description())
    
    def get_output_format_description(self) -> str:
        """返回输出格式说明"""
        return getattr(self, '_output_format', super().get_output_format_description())
    
    def get_simulator_description(self) -> str:
        """返回模拟器说明"""
        return getattr(self, '_simulator', super().get_simulator_description())
    
    def get_strategy_description(self) -> str:
        """返回策略说明"""
        return getattr(self, '_strategy', super().get_strategy_description())
    
    def get_experiment_setting_description(self) -> str:
        """返回实验设置说明"""
        return getattr(self, '_experiment_setting', super().get_experiment_setting_description())
    
    def get_runtime_environment(self) -> str:
        """获取涨停板运行环境信息"""
        try:
            # 尝试导入数据接口
            import sys
            import importlib.util
            
            env_info = f"""
涨停板数据接口：LimitUpDataInterface
可用字段：seal_amount, continuous_board, concept_heat, limit_up_minutes, 
          large_buy, large_sell, open_count, seal_quality
Python 版本：{sys.version.split()[0]}
            """
            
            # 检查 LimitUpDataInterface 是否可用
            try:
                from rd_agent.limit_up_data import LimitUpDataInterface
                env_info += "\n✅ LimitUpDataInterface 可用"
            except ImportError:
                env_info += "\n⚠️ LimitUpDataInterface 未找到，请检查 rd_agent/limit_up_data.py"
            
            return env_info.strip()
            
        except Exception as e:
            return f"运行环境信息获取失败: {e}"


# 便捷工厂函数
def create_limitup_scenario(**kwargs) -> LimitUpFactorScenario:
    """
    创建涨停板因子场景实例
    
    Args:
        **kwargs: 传递给 LimitUpFactorScenario 的参数
    
    Returns:
        LimitUpFactorScenario: 涨停板因子场景实例
    
    Example:
        >>> scenario = create_limitup_scenario()
        >>> print(scenario.background)
    """
    return LimitUpFactorScenario(**kwargs)


if __name__ == "__main__":
    # 测试场景创建
    print("=" * 60)
    print("LimitUpFactorScenario 测试")
    print("=" * 60)
    
    try:
        scenario = create_limitup_scenario()
        print("\n✅ 场景创建成功")
        print(f"\n场景背景（前 200 字符）：\n{scenario.background[:200]}...")
        print(f"\n数据源说明（前 200 字符）：\n{scenario.get_source_data_desc()[:200]}...")
        print(f"\n运行环境：\n{scenario.get_runtime_environment()}")
    except Exception as e:
        print(f"\n❌ 场景创建失败: {e}")
        import traceback
        traceback.print_exc()
