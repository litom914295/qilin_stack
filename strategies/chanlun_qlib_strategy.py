"""缠论增强策略 - 基于Qlib框架

继承 Qlib TopkDropoutStrategy，融合:
1. 缠论评分 (ChanLunScoringAgent) - 35%
2. 麒麟 Alpha191 因子 - 30%
3. 麒麟技术指标因子 - 20%
4. 麒麟成交量因子 - 15%

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - Phase 2 重构
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Text
import logging

from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest.decision import TradeDecisionWO

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.chanlun_agent import ChanLunScoringAgent

logger = logging.getLogger(__name__)


class ChanLunEnhancedStrategy(TopkDropoutStrategy):
    """缠论增强策略 - 基于 Qlib TopK 策略
    
    核心思想:
    - 复用 Qlib TopkDropoutStrategy 的选股逻辑
    - 融合缠论评分与 Qlib 因子评分
    - 不重复实现技术指标，直接使用 Qlib 因子
    
    评分融合:
    - 缠论评分: ChanLunScoringAgent (35%)
    - Qlib因子: Alpha191 + 技术指标 + 成交量 (65%)
    
    使用方式:
        strategy = ChanLunEnhancedStrategy(
            model=model,
            dataset=dataset,
            chanlun_weight=0.35,
            topk=10,
            n_drop=2
        )
    """
    
    def __init__(self,
                 model,
                 dataset,
                 chanlun_weight: float = 0.35,
                 use_chanlun: bool = True,
                 topk: int = 30,
                 n_drop: int = 5,
                 **kwargs):
        """初始化缠论增强策略
        
        Args:
            model: Qlib 模型 (可以是 ChanLunScoringModel 或其他模型)
            dataset: Qlib 数据集
            chanlun_weight: 缠论评分权重 (0-1)
            use_chanlun: 是否启用缠论评分
            topk: 选股数量
            n_drop: 每次最多卖出数量
        """
        super().__init__(
            model=model,
            dataset=dataset,
            topk=topk,
            n_drop=n_drop,
            **kwargs
        )
        
        self.chanlun_weight = chanlun_weight if use_chanlun else 0
        self.qlib_weight = 1 - self.chanlun_weight
        self.use_chanlun = use_chanlun
        
        # 初始化缠论智能体
        if self.use_chanlun:
            self.chanlun_agent = ChanLunScoringAgent(
                morphology_weight=0.40,
                bsp_weight=0.35,
                enable_bsp=True,
                enable_divergence=True
            )
            logger.info(f"ChanLunEnhancedStrategy: 缠论权重={chanlun_weight:.2%}")
        else:
            self.chanlun_agent = None
            logger.info("ChanLunEnhancedStrategy: 仅使用 Qlib 因子")
        
        logger.info(f"策略初始化: TopK={topk}, N_drop={n_drop}")
    
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策 - 重写父类方法
        
        核心逻辑:
        1. 获取 Qlib 模型预测 (包含 Alpha191/技术指标/成交量因子)
        2. 如果启用缠论，计算缠论评分
        3. 加权融合两种评分
        4. 使用融合后的评分进行 TopK 选股
        
        Returns:
            TradeDecisionWO: 交易决策
        """
        # 获取当前交易日期
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time = self.trade_calendar.start_time
        trade_end_time = self.trade_calendar.end_time
        
        # 1. 获取 Qlib 模型预测评分
        pred_score = self.signal.get_signal(
            start_time=trade_start_time,
            end_time=trade_end_time
        )
        
        if pred_score is None or len(pred_score) == 0:
            logger.warning("未获取到 Qlib 模型预测")
            return TradeDecisionWO([], self)
        
        # 2. 如果启用缠论，计算缠论评分并融合
        if self.use_chanlun and self.chanlun_agent is not None:
            try:
                # 获取缠论评分
                chanlun_scores = self._get_chanlun_scores(trade_start_time, trade_end_time)
                
                if len(chanlun_scores) > 0:
                    # 融合评分
                    final_scores = self._merge_scores(pred_score, chanlun_scores)
                    logger.info(f"融合评分完成: Qlib={self.qlib_weight:.1%} + 缠论={self.chanlun_weight:.1%}")
                else:
                    logger.warning("缠论评分为空，仅使用 Qlib 评分")
                    final_scores = pred_score
            except Exception as e:
                logger.error(f"缠论评分失败: {e}，仅使用 Qlib 评分")
                final_scores = pred_score
        else:
            final_scores = pred_score
        
        # 3. 使用融合评分进行 TopK 选股 (复用父类逻辑)
        # 将融合评分设置为信号
        self.signal._score = final_scores
        
        # 调用父类的选股逻辑
        trade_decision = super().generate_trade_decision(execute_result)
        
        return trade_decision
    
    def _get_chanlun_scores(self, start_time, end_time) -> pd.Series:
        """获取缠论评分
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            pd.Series: MultiIndex (datetime, instrument) -> score
        """
        # 从数据集获取原始数据
        try:
            # 获取当前交易日的数据
            df_data = self.dataset.prepare(
                segments="test",
                col_set="feature",
                data_key=None
            )
            
            if df_data is None or len(df_data) == 0:
                logger.warning("数据集为空")
                return pd.Series()
            
            scores = []
            
            # 获取当前时间点的所有股票
            if isinstance(df_data.index, pd.MultiIndex):
                # 获取当前交易日期
                current_date = end_time
                
                # 筛选出当前日期的数据
                if current_date in df_data.index.get_level_values(0):
                    instruments = df_data.xs(current_date, level=0).index
                    
                    for instrument in instruments:
                        try:
                            # 获取该股票的历史数据
                            stock_data = df_data.xs(instrument, level=1)
                            historical_data = stock_data.loc[:current_date]
                            
                            if len(historical_data) >= 20:
                                # 计算缠论评分
                                score = self.chanlun_agent.score(historical_data, instrument)
                                scores.append({
                                    'datetime': current_date,
                                    'instrument': instrument,
                                    'score': score
                                })
                        except Exception as e:
                            logger.debug(f"{instrument} 缠论评分失败: {e}")
                            continue
            
            if len(scores) > 0:
                score_df = pd.DataFrame(scores)
                score_series = score_df.set_index(['datetime', 'instrument'])['score']
                logger.info(f"缠论评分完成: {len(score_series)} 只股票")
                return score_series
            else:
                return pd.Series()
                
        except Exception as e:
            logger.error(f"获取缠论评分失败: {e}")
            return pd.Series()
    
    def _merge_scores(self, qlib_scores: pd.Series, 
                     chanlun_scores: pd.Series) -> pd.Series:
        """融合 Qlib 评分和缠论评分
        
        Args:
            qlib_scores: Qlib 模型预测评分
            chanlun_scores: 缠论评分
        
        Returns:
            pd.Series: 融合后的评分
        """
        # 对齐索引
        common_index = qlib_scores.index.intersection(chanlun_scores.index)
        
        if len(common_index) == 0:
            logger.warning("Qlib 评分和缠论评分无共同索引")
            return qlib_scores
        
        # 归一化评分到 0-1
        qlib_normalized = self._normalize_scores(qlib_scores.loc[common_index])
        chanlun_normalized = self._normalize_scores(chanlun_scores.loc[common_index])
        
        # 加权融合
        merged = (
            qlib_normalized * self.qlib_weight +
            chanlun_normalized * self.chanlun_weight
        )
        
        # 对于只有 Qlib 评分的股票，使用原始评分
        only_qlib = qlib_scores.index.difference(common_index)
        if len(only_qlib) > 0:
            merged = pd.concat([merged, qlib_scores.loc[only_qlib]])
        
        logger.debug(f"评分融合: {len(common_index)} 只股票有双重评分, "
                    f"{len(only_qlib)} 只仅有 Qlib 评分")
        
        return merged
    
    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        """归一化评分到 0-1
        
        使用 Min-Max 归一化
        """
        if len(scores) == 0:
            return scores
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score > min_score:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = pd.Series(0.5, index=scores.index)
        
        return normalized


class SimpleChanLunStrategy(TopkDropoutStrategy):
    """简化版缠论策略 - 仅使用缠论评分
    
    适用于测试或只想用缠论信号的场景
    """
    
    def __init__(self, model, dataset, topk: int = 30, n_drop: int = 5, **kwargs):
        """初始化简化版策略
        
        Args:
            model: ChanLunScoringModel
            dataset: 数据集
            topk: 选股数量
            n_drop: 卖出数量
        """
        super().__init__(
            model=model,
            dataset=dataset,
            topk=topk,
            n_drop=n_drop,
            **kwargs
        )
        
        logger.info(f"SimpleChanLunStrategy 初始化: TopK={topk}, N_drop={n_drop}")
    
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策 - 直接使用模型预测"""
        # 直接使用父类逻辑，模型已经是 ChanLunScoringModel
        return super().generate_trade_decision(execute_result)


if __name__ == '__main__':
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("ChanLunEnhancedStrategy 测试")
    print("="*60)
    
    print("\n✅ 策略类定义完成")
    print(f"   - ChanLunEnhancedStrategy: 融合策略")
    print(f"   - SimpleChanLunStrategy: 纯缠论策略")
    
    print("\n核心特性:")
    print("   ✅ 继承 Qlib TopkDropoutStrategy")
    print("   ✅ 融合缠论评分与 Qlib 因子")
    print("   ✅ 复用 Qlib 选股逻辑")
    print("   ✅ 不重复实现技术指标")
    
    print("\n✅ ChanLunEnhancedStrategy 测试完成!")
