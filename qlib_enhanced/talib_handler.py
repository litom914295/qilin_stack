#!/usr/bin/env python
"""
Qlib DataHandler集成TA-Lib
支持在Qlib工作流中使用TA-Lib技术指标和缠论形态特征
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import processor as processor_module

# 导入TA-Lib包装器
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from features.talib_indicators import TALibFeatureGenerator


class TALibHandler(DataHandlerLP):
    """
    集成TA-Lib的Qlib DataHandler
    
    使用方法:
        在Qlib配置文件中:
        handler:
            class: TALibHandler
            module_path: qlib_enhanced.talib_handler
            kwargs:
                start_time: "2010-01-01"
                end_time: "2020-12-31"
                instruments: "csi300"
                include_patterns: true  # 是否包含K线形态
                feature_groups:  # 可选特征组
                    - trend      # 趋势指标
                    - momentum   # 动量指标
                    - volatility # 波动率指标
                    - volume     # 成交量指标
                    - patterns   # K线形态
    """
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        include_patterns: bool = True,
        feature_groups: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            include_patterns: 是否包含K线形态特征
            feature_groups: 特征组列表，可选值: ['trend', 'momentum', 'volatility', 'volume', 'patterns']
                          如果为None，则使用所有特征
        """
        self.include_patterns = include_patterns
        self.feature_groups = feature_groups
        self.talib_generator = TALibFeatureGenerator(include_patterns=include_patterns)
        
        # 定义数据字段
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self._get_qlib_fields(),
                "freq": freq,
                "filter_pipe": filter_pipe,
            },
        }
        
        # 默认处理器
        if not infer_processors:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
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
    
    def _get_qlib_fields(self):
        """获取需要从Qlib加载的原始字段"""
        fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
        
        # 添加标签
        fields.append("Ref($close, -2)/Ref($close, -1) - 1")  # 未来1日收益率
        
        names = ["open", "high", "low", "close", "volume", "factor"]
        names.append("LABEL0")
        
        return fields, names
    
    def _apply_talib_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        在Qlib数据上应用TA-Lib特征
        
        Args:
            df: Qlib加载的原始数据
            
        Returns:
            包含TA-Lib特征的DataFrame
        """
        # 按股票分组计算TA-Lib特征
        result_list = []
        
        for instrument, group in df.groupby(level='instrument'):
            try:
                # 生成TA-Lib特征
                talib_features = self.talib_generator.generate_features(group)
                
                # 过滤特征组
                if self.feature_groups is not None:
                    talib_features = self._filter_feature_groups(talib_features)
                
                # 添加instrument索引
                talib_features['instrument'] = instrument
                talib_features['datetime'] = group.index.get_level_values('datetime')
                talib_features.set_index(['instrument', 'datetime'], inplace=True)
                
                result_list.append(talib_features)
            except Exception as e:
                print(f"Warning: Failed to calculate TA-Lib features for {instrument}: {e}")
                continue
        
        if result_list:
            return pd.concat(result_list, axis=0)
        else:
            return pd.DataFrame()
    
    def _filter_feature_groups(self, features: pd.DataFrame) -> pd.DataFrame:
        """根据feature_groups过滤特征"""
        if self.feature_groups is None:
            return features
        
        # 定义特征组
        feature_map = {
            'trend': ['sma_', 'ema_', 'macd', 'adx'],
            'momentum': ['rsi_', 'stoch_', 'cci_', 'mom_', 'roc_', 'willr_'],
            'volatility': ['atr_', 'natr_', 'bbands_'],
            'volume': ['obv', 'ad', 'mfi_'],
            'patterns': ['doji', 'hammer', 'engulfing', 'harami', 'star', 'soldiers', 'crows', 'bi_', 'duan_']
        }
        
        selected_cols = []
        for group in self.feature_groups:
            if group in feature_map:
                patterns = feature_map[group]
                for col in features.columns:
                    if any(pattern in col for pattern in patterns):
                        selected_cols.append(col)
        
        return features[selected_cols] if selected_cols else features
    
    def fetch(self, selector=None, level=None, col_set=None):
        """重写fetch方法，添加TA-Lib特征"""
        # 调用父类fetch获取Qlib数据
        df = super().fetch(selector=selector, level=level, col_set=col_set)
        
        if df.empty:
            return df
        
        # 应用TA-Lib特征
        talib_df = self._apply_talib_features(df)
        
        if not talib_df.empty:
            # 合并TA-Lib特征与原始数据
            df = df.join(talib_df, how='left')
        
        return df


class TALibAlpha360Handler(DataHandlerLP):
    """
    Alpha360 + TA-Lib 混合Handler
    结合Qlib内置的Alpha360特征和TA-Lib特征
    """
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        include_patterns: bool = False,  # 默认不包含形态，因为Alpha360已经很多特征
        **kwargs
    ):
        self.include_patterns = include_patterns
        self.talib_generator = TALibFeatureGenerator(include_patterns=include_patterns)
        
        # 使用Alpha360的字段配置
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self._get_alpha360_fields(),
                "freq": freq,
                "filter_pipe": filter_pipe,
            },
        }
        
        if not infer_processors:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
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
    
    def _get_alpha360_fields(self):
        """获取Alpha360字段（简化版，实际应该从Alpha360继承）"""
        # 这里简化处理，实际使用时可以从qlib.contrib.data.handler.Alpha360继承
        fields = []
        names = []
        
        # 基础价格特征
        for i in [0, 1, 2, 3, 4]:
            fields.append(f"Ref($close, {i})/$close")
            names.append(f"CLOSE{i}")
        
        # 添加TA-Lib相关的OHLCV
        fields.extend(["$open", "$high", "$low", "$close", "$volume"])
        names.extend(["open", "high", "low", "close", "volume"])
        
        # 标签
        fields.append("Ref($close, -2)/Ref($close, -1) - 1")
        names.append("LABEL0")
        
        return fields, names
    
    def fetch(self, selector=None, level=None, col_set=None):
        """重写fetch，添加TA-Lib特征"""
        df = super().fetch(selector=selector, level=level, col_set=col_set)
        
        if df.empty:
            return df
        
        # 提取OHLCV数据计算TA-Lib特征
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in ohlcv_cols):
            talib_df = self._apply_talib_features(df[ohlcv_cols])
            if not talib_df.empty:
                df = df.join(talib_df, how='left')
        
        return df
    
    def _apply_talib_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用TA-Lib特征（复用TALibHandler的逻辑）"""
        result_list = []
        
        for instrument, group in df.groupby(level='instrument'):
            try:
                talib_features = self.talib_generator.generate_features(group)
                talib_features['instrument'] = instrument
                talib_features['datetime'] = group.index.get_level_values('datetime')
                talib_features.set_index(['instrument', 'datetime'], inplace=True)
                result_list.append(talib_features)
            except Exception as e:
                print(f"Warning: TA-Lib features failed for {instrument}: {e}")
                continue
        
        if result_list:
            return pd.concat(result_list, axis=0)
        else:
            return pd.DataFrame()


class LimitUpTALibHandler(TALibHandler):
    """
    涨停板专用TA-Lib Handler
    针对一进二场景优化，包含缠论形态特征
    """
    
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        **kwargs
    ):
        # 强制包含K线形态
        kwargs['include_patterns'] = True
        
        # 涨停板重点关注的特征组
        kwargs['feature_groups'] = ['trend', 'momentum', 'volatility', 'volume', 'patterns']
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            **kwargs
        )
    
    def _get_qlib_fields(self):
        """涨停板特殊字段"""
        fields, names = super()._get_qlib_fields()
        
        # 添加涨停板相关特征
        fields.extend([
            "($close - $open) / $open",  # 涨幅
            "$volume / Mean($volume, 5)",  # 量比
            "($high - $low) / $open",  # 振幅
        ])
        names.extend(["pct_chg", "volume_ratio", "amplitude"])
        
        # 涨停板标签：今日涨停 AND 明日续涨
        fields[-1] = "(Ref($close, -1) > $close * 1.095) & (Ref($close, -2) > Ref($close, -1))"
        names[-1] = "LABEL_LIMITUP"
        
        return fields, names


if __name__ == '__main__':
    print("Qlib TA-Lib Handler 集成模块")
    print("=" * 60)
    print("✅ TALibHandler - 纯TA-Lib特征")
    print("✅ TALibAlpha360Handler - Alpha360 + TA-Lib混合")
    print("✅ LimitUpTALibHandler - 涨停板专用（含缠论形态）")
    print("=" * 60)
    print("\n使用示例:")
    print("""
# 在Qlib配置文件中使用:
task:
    dataset:
        handler:
            class: TALibHandler
            module_path: qlib_enhanced.talib_handler
            kwargs:
                start_time: "2010-01-01"
                end_time: "2020-12-31"
                instruments: "csi300"
                include_patterns: true
                feature_groups: ["trend", "momentum", "patterns"]
    """)
