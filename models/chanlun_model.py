"""缠论评分模型 - Qlib接口适配

将缠论评分智能体适配为Qlib标准Model接口，用于回测系统

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - 缠论Qlib集成
"""

import pandas as pd
import numpy as np
from typing import Union, Text, Optional
import logging

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.chanlun_agent import ChanLunScoringAgent

logger = logging.getLogger(__name__)


class ChanLunScoringModel(Model):
    """缠论评分模型 - Qlib标准Model接口
    
    功能:
    1. 包装 ChanLunScoringAgent 为 Qlib Model
    2. 实现 fit() 和 predict() 接口
    3. 支持 Qlib 回测系统集成
    
    使用方式:
        model = ChanLunScoringModel(
            morphology_weight=0.40,
            bsp_weight=0.35
        )
        model.fit(dataset)
        predictions = model.predict(dataset)
    """
    
    def __init__(self, 
                 morphology_weight=0.40,
                 bsp_weight=0.35,
                 divergence_weight=0.15,
                 multi_level_weight=0.10,
                 enable_bsp=True,
                 enable_divergence=True,
                 use_multi_level=False,
                 **kwargs):
        """初始化缠论评分模型
        
        Args:
            morphology_weight: 形态评分权重
            bsp_weight: 买卖点评分权重
            divergence_weight: 背驰评分权重
            multi_level_weight: 多级别共振权重
            enable_bsp: 是否启用买卖点评分
            enable_divergence: 是否启用背驰评分
            use_multi_level: 是否启用多级别共振
        """
        super().__init__(**kwargs)
        
        # 初始化缠论智能体
        self.agent = ChanLunScoringAgent(
            morphology_weight=morphology_weight,
            bsp_weight=bsp_weight,
            divergence_weight=divergence_weight,
            multi_level_weight=multi_level_weight,
            enable_bsp=enable_bsp,
            enable_divergence=enable_divergence,
            use_multi_level=use_multi_level
        )
        
        logger.info(f"ChanLunScoringModel 初始化: "
                   f"形态{morphology_weight:.2f} 买卖点{bsp_weight:.2f} "
                   f"背驰{divergence_weight:.2f} 多级别{multi_level_weight:.2f}")
    
    def fit(self, dataset: DatasetH, **kwargs):
        """训练模型 (缠论评分无需训练)
        
        缠论评分是基于规则的评分系统，不需要训练过程。
        此方法仅为兼容Qlib接口而保留。
        
        Args:
            dataset: Qlib DatasetH 对象
        """
        logger.info("ChanLunScoringModel 无需训练 (规则评分系统)")
        # 缠论评分是规则系统，无需训练
        pass
    
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测股票评分
        
        对数据集中的所有股票进行缠论评分
        
        Args:
            dataset: Qlib DatasetH 对象
            segment: 数据切片 ('train', 'valid', 'test')
        
        Returns:
            pd.Series: 多级索引 (datetime, instrument) -> score
        """
        logger.info(f"开始缠论评分预测: segment={segment}")
        
        # 获取数据
        if segment == "test":
            df_data = dataset.prepare("test", col_set="feature")
        elif segment == "valid":
            df_data = dataset.prepare("valid", col_set="feature")
        else:
            df_data = dataset.prepare("train", col_set="feature")
        
        if df_data is None or len(df_data) == 0:
            logger.warning("数据集为空")
            return pd.Series()
        
        logger.info(f"数据集大小: {df_data.shape}")
        
        # 对每个股票每个时间点评分
        predictions = []
        
        # 获取所有股票代码
        if isinstance(df_data.index, pd.MultiIndex):
            # MultiIndex: (datetime, instrument)
            instruments = df_data.index.get_level_values(1).unique()
            
            for instrument in instruments:
                try:
                    # 获取该股票的所有数据
                    stock_data = df_data.xs(instrument, level=1)
                    
                    if len(stock_data) < 20:
                        logger.debug(f"{instrument}: 数据不足20条，跳过")
                        continue
                    
                    # 对每个时间点评分
                    for timestamp in stock_data.index:
                        # 获取截止到该时间点的历史数据
                        historical_data = stock_data.loc[:timestamp]
                        
                        if len(historical_data) >= 20:
                            # 计算缠论评分
                            score = self.agent.score(historical_data, instrument)
                            predictions.append({
                                'datetime': timestamp,
                                'instrument': instrument,
                                'score': score
                            })
                
                except Exception as e:
                    logger.error(f"{instrument} 评分失败: {e}")
                    continue
        else:
            logger.error("数据集索引格式不正确，需要 MultiIndex (datetime, instrument)")
            return pd.Series()
        
        # 转换为 Series
        if len(predictions) > 0:
            pred_df = pd.DataFrame(predictions)
            pred_series = pred_df.set_index(['datetime', 'instrument'])['score']
            logger.info(f"预测完成: {len(pred_series)} 条评分")
            return pred_series
        else:
            logger.warning("没有生成任何预测")
            return pd.Series()
    
    def finetune(self, dataset: DatasetH, **kwargs):
        """微调模型 (缠论评分无需微调)
        
        Args:
            dataset: Qlib DatasetH 对象
        """
        logger.info("ChanLunScoringModel 无需微调")
        pass


class ChanLunSignalModel(ChanLunScoringModel):
    """缠论信号模型 - 输出买卖信号而非评分
    
    继承自 ChanLunScoringModel，但输出转换为:
    - 1: 买入信号 (评分 >= 75)
    - 0: 持有/观望 (60 <= 评分 < 75)
    - -1: 卖出信号 (评分 < 60)
    """
    
    def __init__(self, 
                 buy_threshold=75.0,
                 sell_threshold=60.0,
                 **kwargs):
        """初始化信号模型
        
        Args:
            buy_threshold: 买入阈值 (评分>=此值时买入)
            sell_threshold: 卖出阈值 (评分<此值时卖出)
        """
        super().__init__(**kwargs)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        logger.info(f"ChanLunSignalModel 初始化: "
                   f"买入阈值={buy_threshold} 卖出阈值={sell_threshold}")
    
    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测买卖信号
        
        Returns:
            pd.Series: 1 (买入), 0 (持有), -1 (卖出)
        """
        # 获取评分
        scores = super().predict(dataset, segment)
        
        if len(scores) == 0:
            return pd.Series()
        
        # 转换为信号
        signals = pd.Series(0, index=scores.index)  # 默认持有
        signals[scores >= self.buy_threshold] = 1   # 买入
        signals[scores < self.sell_threshold] = -1  # 卖出
        
        logger.info(f"信号分布: 买入{(signals==1).sum()} "
                   f"持有{(signals==0).sum()} 卖出{(signals==-1).sum()}")
        
        return signals


if __name__ == '__main__':
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("ChanLunScoringModel 测试")
    print("="*60)
    
    # 创建模型
    model = ChanLunScoringModel(
        morphology_weight=0.40,
        bsp_weight=0.35,
        enable_bsp=True
    )
    
    print("\n✅ 模型创建成功")
    print(f"   类型: {type(model)}")
    print(f"   智能体: {type(model.agent)}")
    
    # 创建信号模型
    signal_model = ChanLunSignalModel(
        buy_threshold=75.0,
        sell_threshold=60.0
    )
    
    print("\n✅ 信号模型创建成功")
    print(f"   买入阈值: {signal_model.buy_threshold}")
    print(f"   卖出阈值: {signal_model.sell_threshold}")
    
    print("\n✅ ChanLunScoringModel 测试完成!")
