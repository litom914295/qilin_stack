"""简化的缠论 Handler - 基于因子注册

特点:
- 不包含特征生成逻辑，仅加载已注册的因子
- 与 Qlib 因子体系完全兼容
- 代码简洁，易于维护

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - Phase 3 优化
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qlib.data.dataset.handler import DataHandlerLP
from qlib_enhanced.chanlun.register_factors import (
    register_chanlun_factors,
    get_factor_names,
    compute_all_factors
)
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ChanLunFactorHandler(DataHandlerLP):
    """缠论因子Handler - 简化版
    
    特点:
    - 自动注册缠论因子
    - 仅作为因子加载器，不包含特征生成逻辑
    - 与 Qlib 因子库完全兼容
    
    使用方法:
        在 Qlib 配置文件中:
        handler:
            class: ChanLunFactorHandler
            module_path: qlib_enhanced.chanlun.factor_handler
            kwargs:
                start_time: "2020-01-01"
                end_time: "2023-12-31"
                instruments: "csi300"
                use_czsc: true          # 是否使用 CZSC 因子
                use_chanpy: true        # 是否使用 Chan.py 因子
                drop_raw: false         # 是否删除原始 OHLCV
    """
    
    def __init__(self,
                 instruments='csi300',
                 start_time=None,
                 end_time=None,
                 freq='day',
                 infer_processors=[],
                 learn_processors=[],
                 fit_start_time=None,
                 fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A,
                 use_czsc=True,
                 use_chanpy=True,
                 drop_raw=False,
                 **kwargs):
        """初始化缠论因子Handler
        
        Args:
            use_czsc: 是否加载 CZSC 因子
            use_chanpy: 是否加载 Chan.py 因子
            drop_raw: 是否删除原始 OHLCV 数据
        """
        
        self.use_czsc = use_czsc
        self.use_chanpy = use_chanpy
        self.drop_raw = drop_raw
        self.freq = freq
        
        # 注册缠论因子
        register_chanlun_factors()
        
        # 确定要加载的因子
        self.chanlun_factors = []
        if use_czsc:
            self.chanlun_factors.extend(get_factor_names('czsc'))
        if use_chanpy:
            self.chanlun_factors.extend(get_factor_names('chanpy'))
        
        logger.info(f"ChanLunFactorHandler: 将加载 {len(self.chanlun_factors)} 个缠论因子")
        
        # 定义数据加载器
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self._get_base_fields(),
                "freq": freq,
            },
        }
        
        # 默认处理器
        if not infer_processors:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {
                    "fields_group": "feature", 
                    "clip_outlier": True,
                    "fit_start_time": fit_start_time,
                    "fit_end_time": fit_end_time
                }},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
        
        if not learn_processors:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ]
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )
    
    def _get_base_fields(self):
        """定义基础字段"""
        fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
        names = ["open", "close", "high", "low", "volume", "factor"]
        
        # 添加标签
        fields.append("Ref($close, -2)/Ref($close, -1) - 1")
        names.append("LABEL0")
        
        return fields, names
    
    def fetch(self, selector=None, level=None, col_set=None):
        """重写 fetch 方法，添加缠论因子
        
        注意: 这里的实现是计算缠论因子，而非从 Qlib 因子库加载
        因为 Qlib 的因子注册需要特殊的表达式引擎支持
        当前实现作为过渡方案，保持与原 Handler 相同的接口
        """
        # 1. 获取基础 OHLCV 数据
        df = super().fetch(selector=selector, level=level, col_set=col_set)
        
        if df is None or len(df) == 0:
            logger.warning("基础数据为空")
            return df
        
        logger.info(f"开始添加缠论因子, 股票数: {len(df.index.get_level_values(0).unique())}")
        
        # 2. 按股票分组计算因子
        factor_list = []
        
        for instrument in df.index.get_level_values(0).unique():
            try:
                inst_df = df.loc[instrument].reset_index()
                
                # 准备输入格式
                input_df = pd.DataFrame({
                    'datetime': inst_df['datetime'],
                    'open': inst_df['open'],
                    'close': inst_df['close'],
                    'high': inst_df['high'],
                    'low': inst_df['low'],
                    'volume': inst_df['volume'],
                })
                
                # 计算缠论因子
                category = None
                if self.use_czsc and self.use_chanpy:
                    category = None  # 计算所有因子
                elif self.use_czsc:
                    category = 'czsc'
                elif self.use_chanpy:
                    category = 'chanpy'
                
                result = compute_all_factors(input_df, code=instrument, category=category)
                result['instrument'] = instrument
                result['datetime'] = inst_df['datetime'].values
                
                factor_list.append(result)
                
            except Exception as e:
                logger.error(f"股票 {instrument} 因子计算失败: {e}")
                continue
        
        if not factor_list:
            logger.warning("无缠论因子生成")
            return df
        
        # 3. 合并因子
        factor_df = pd.concat(factor_list, ignore_index=True)
        factor_df = factor_df.set_index(['instrument', 'datetime'])
        
        # 4. 添加因子列到 DataFrame
        for col in self.chanlun_factors:
            if col in factor_df.columns:
                df[col] = factor_df[col]
        
        # 5. 可选: 删除原始 OHLCV
        if self.drop_raw:
            df = df.drop(columns=['open', 'high', 'low'], errors='ignore')
        
        logger.info(f"✅ 缠论因子加载完成, 共 {len(self.chanlun_factors)} 个因子")
        
        return df


if __name__ == '__main__':
    # 简单测试 - 仅测试因子注册，不加载实际数据
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("ChanLunFactorHandler 测试")
    print("="*60)
    
    # 测试因子注册
    register_chanlun_factors()
    czsc_factors = get_factor_names('czsc')
    chanpy_factors = get_factor_names('chanpy')
    
    print(f"\n✅ 缠论因子注册成功")
    print(f"   CZSC 因子: {len(czsc_factors)} 个")
    print(f"   Chan.py 因子: {len(chanpy_factors)} 个")
    print(f"   总计: {len(czsc_factors) + len(chanpy_factors)} 个")
    
    print("\n📝 CZSC 因子列表:")
    for name in czsc_factors:
        print(f"   - {name}")
    
    print("\n📝 Chan.py 因子列表:")
    for name in chanpy_factors:
        print(f"   - {name}")
    
    print("\nℹ️  注意: Handler 的完整测试需要先初始化 Qlib (qlib.init())")
    print("   在实际使用中，通过 Qlib 配置文件加载 Handler 即可")
    
    print("\n✅ ChanLunFactorHandler 测试完成!")
