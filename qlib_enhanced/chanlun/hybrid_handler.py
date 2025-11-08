"""混合Handler: CZSC + Chan.py"""

from qlib_enhanced.chanlun.czsc_handler import CzscChanLunHandler
from features.chanlun.chanpy_features import ChanPyFeatureGenerator
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class HybridChanLunHandler(CzscChanLunHandler):
    """
    混合缠论Handler
    
    策略:
    - CZSC: 快速形态识别 (分型/笔/中枢基础)
    - Chan.py: 买卖点识别 (6类买卖点/线段/完整中枢)
    - 结果融合: 16个特征 (6个CZSC + 10个Chan.py)
    
    使用方法:
        在Qlib配置文件中:
        handler:
            class: HybridChanLunHandler
            module_path: qlib_enhanced.chanlun.hybrid_handler
            kwargs:
                start_time: "2020-01-01"
                end_time: "2023-12-31"
                instruments: "csi300"
                use_chanpy: true        # 是否使用Chan.py
                seg_algo: "chan"        # 线段算法
    """
    
    def __init__(self, 
                 use_chanpy=True,
                 seg_algo='chan',
                 bi_algo='normal',
                 zs_combine=True,
                 **kwargs):
        """
        Args:
            use_chanpy: 是否使用Chan.py买卖点
            seg_algo: 线段算法 ('chan'/'def'/'dyh')
            bi_algo: 笔算法
            zs_combine: 是否合并中枢
            **kwargs: 传递给父类CzscChanLunHandler的参数
        """
        self.use_chanpy = use_chanpy
        
        # 初始化Chan.py生成器
        if use_chanpy:
            self.chanpy_gen = ChanPyFeatureGenerator(
                seg_algo=seg_algo,
                bi_algo=bi_algo,
                zs_combine=zs_combine
            )
        
        super().__init__(**kwargs)
    
    def fetch(self, selector=None, level=None, col_set=None):
        """重写fetch方法, 添加Chan.py特征"""
        # 1. 获取CZSC特征 (调用父类)
        df = super().fetch(selector=selector, level=level, col_set=col_set)
        
        if not self.use_chanpy or df is None or len(df) == 0:
            return df
        
        logger.info("开始计算Chan.py买卖点特征...")
        
        # 2. 按股票分组添加Chan.py特征
        chanpy_features_list = []
        
        for instrument in df.index.get_level_values(0).unique():
            try:
                inst_df = df.loc[instrument].reset_index()
                
                # 准备Chan.py输入 (需要处理Qlib的字段名)
                chanpy_input = pd.DataFrame({
                    'datetime': inst_df['datetime'],
                    'open': inst_df.get('open', inst_df.get('$open', 0)),
                    'close': inst_df.get('close', inst_df.get('$close', 0)),
                    'high': inst_df.get('high', inst_df.get('$high', 0)),
                    'low': inst_df.get('low', inst_df.get('$low', 0)),
                    'volume': inst_df.get('volume', inst_df.get('$volume', 0)),
                })
                
                # 生成Chan.py特征
                chanpy_result = self.chanpy_gen.generate_features(chanpy_input, code=instrument)
                chanpy_result['instrument'] = instrument
                chanpy_result['datetime'] = inst_df['datetime'].values
                
                chanpy_features_list.append(chanpy_result)
                
            except Exception as e:
                logger.error(f"股票{instrument} Chan.py特征计算失败: {e}")
                continue
        
        if not chanpy_features_list:
            logger.warning("无Chan.py特征生成")
            return df
        
        # 3. 合并Chan.py特征
        chanpy_df = pd.concat(chanpy_features_list, ignore_index=True)
        chanpy_df = chanpy_df.set_index(['instrument', 'datetime'])
        
        # 4. 添加Chan.py特征列到DataFrame
        chanpy_cols = [
            'is_buy_point', 'is_sell_point', 'bsp_type', 'bsp_is_buy',
            'seg_direction', 'is_seg_start', 'is_seg_end',
            'in_chanpy_zs', 'zs_low_chanpy', 'zs_high_chanpy'
        ]
        
        for col in chanpy_cols:
            if col in chanpy_df.columns:
                df[col] = chanpy_df[col]
        
        logger.info(f"✅ Chan.py特征计算完成, 新增特征: {len(chanpy_cols)}")
        logger.info(f"📊 混合Handler总特征数: CZSC(6) + Chan.py({len(chanpy_cols)}) = {6 + len(chanpy_cols)}")
        
        return df
