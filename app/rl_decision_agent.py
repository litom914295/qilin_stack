"""
麒麟量化系统 - 强化学习+自我进化智能体选股决策引擎
基于集合竞价数据进行智能选股排序
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class StockFeatures:
    """股票特征向量(增强版)"""
    # 昨日信息 (基础)
    consecutive_days: float      # 连板天数
    seal_ratio: float            # 封单强度
    quality_score: float         # 涨停质量分
    is_leader: float            # 是否龙头 (0/1)
    
    # 竞价信息 (基础)
    auction_change: float        # 竞价涨幅
    auction_strength: float      # 竞价强度
    bid_ask_ratio: float        # 买卖比
    large_ratio: float          # 大单占比
    stability: float            # 价格稳定性
    
    # 新增: 分时特征
    vwap_slope: float           # 早盘VWAP斜率
    max_drawdown: float         # 早盘最大回撤
    afternoon_strength: float   # 午后强度
    
    # 新增: 板块热度
    sector_heat: float          # 板块热度 (0-1)
    sector_count: float         # 板块涨停数
    
    # 新增: 首板/连板标识
    is_first_board: float       # 是否首板 (0/1)
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量(16维)"""
        return np.array([
            # 基础特征 (9维)
            self.consecutive_days / 10,  # 归一化
            self.seal_ratio,
            self.quality_score / 100,
            self.is_leader,
            self.auction_change / 10,
            self.auction_strength / 100,
            self.bid_ask_ratio / 5,
            self.large_ratio,
            self.stability / 100,
            # 分时特征 (3维)
            self.vwap_slope * 10 if np.isfinite(self.vwap_slope) else 0,
            abs(self.max_drawdown) * 10 if np.isfinite(self.max_drawdown) else 0,
            self.afternoon_strength * 10 if np.isfinite(self.afternoon_strength) else 0,
            # 板块特征 (2维)
            self.sector_heat,
            self.sector_count / 10,
            # 首板标识 (1维)
            self.is_first_board,
            # 置信度 (1维)
            (self.quality_score / 100 + self.auction_strength / 100) / 2
        ], dtype=np.float32)


class RLDecisionNetwork(nn.Module):
    """强化学习决策网络(增强版)"""
    
    def __init__(self, input_dim: int = 16, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层: 预测得分 (0-100)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 输出 0-1
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"RLDecisionNetwork初始化完成: {input_dim} -> {hidden_dims} -> 1")
    
    def forward(self, x):
        """前向传播"""
        return self.network(x) * 100  # 缩放到 0-100


class SelfEvolutionModule:
    """
    自我进化模块 - 基于历史表现动态调整权重
    新增: Thompson Sampling阈值优化
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # 特征权重 (可学习 - 增强版)
        self.feature_weights = {
            # 基础特征
            "consecutive_days": 0.15,    # 连板天数
            "seal_ratio": 0.12,          # 封单强度
            "quality_score": 0.12,       # 质量分
            "is_leader": 0.08,          # 龙头地位
            "auction_change": 0.12,      # 竞价涨幅
            "auction_strength": 0.12,    # 竞价强度
            "bid_ask_ratio": 0.04,      # 买卖比
            "large_ratio": 0.02,        # 大单占比
            "stability": 0.02,          # 稳定性
            # 分时特征 (新增)
            "vwap_slope": 0.08,         # VWAP斜率
            "max_drawdown": 0.03,       # 最大回撤
            "afternoon_strength": 0.02, # 午后强度
            # 板块特征 (新增)
            "sector_heat": 0.05,        # 板块热度
            "sector_count": 0.02,       # 板块涨停数
            # 首板标识 (新增)
            "is_first_board": 0.05      # 首板加分
        }
        
        # 历史表现记录
        self.performance_history = deque(maxlen=100)
        self.iteration = 0
        
        # Thompson Sampling阈值优化（新增）
        # 动作空间: (min_score, topk) 组合
        self.actions = [
            (60.0, 3), (60.0, 5), (60.0, 10),
            (70.0, 3), (70.0, 5), (70.0, 10),
            (80.0, 3), (80.0, 5), (80.0, 10)
        ]
        # Beta分布参数 {action_key: (alpha, beta)}
        self.bandit_state = {self._action_key(a): [1.0, 1.0] for a in self.actions}
        self.best_action = (70.0, 5)  # 默认推荐
        
        logger.info("自我进化模块初始化完成(含 Thompson Sampling)")
    
    def update_weights(self, stock_symbol: str, predicted_score: float, actual_return: float):
        """
        根据实际收益更新权重
        
        Args:
            stock_symbol: 股票代码
            predicted_score: 预测得分
            actual_return: 实际收益率
        """
        self.performance_history.append({
            "symbol": stock_symbol,
            "predicted": predicted_score,
            "actual": actual_return,
            "iteration": self.iteration
        })
        
        # 每10次更新一次权重
        if len(self.performance_history) >= 10 and self.iteration % 10 == 0:
            self._evolve_weights()
        
        self.iteration += 1
    
    def _evolve_weights(self):
        """进化权重 - 简化版梯度下降"""
        # 计算预测误差
        errors = [
            abs(h["predicted"] - h["actual"] * 100) 
            for h in self.performance_history
        ]
        avg_error = np.mean(errors)
        
        # 如果误差较大,调整权重
        if avg_error > 20:
            # 增加近期表现好的特征权重
            logger.info(f"权重进化: 平均误差 {avg_error:.2f}, 调整权重...")
            
            # 简单策略: 随机微调
            for key in self.feature_weights:
                adjustment = np.random.uniform(-0.02, 0.02) * self.learning_rate
                self.feature_weights[key] = max(0, min(1, self.feature_weights[key] + adjustment))
            
            # 归一化
            total = sum(self.feature_weights.values())
            for key in self.feature_weights:
                self.feature_weights[key] /= total
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.feature_weights.copy()
    
    def save_weights(self, path: str):
        """保存权重和Bandit状态"""
        data = {
            "feature_weights": self.feature_weights,
            "bandit_state": self.bandit_state,
            "best_action": list(self.best_action),
            "iteration": self.iteration
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"权重和Bandit状态已保存: {path}")
    
    def load_weights(self, path: str):
        """加载权重"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.feature_weights = data.get("feature_weights", self.feature_weights)
                # 加载Bandit状态
                if "bandit_state" in data:
                    self.bandit_state = data["bandit_state"]
                if "best_action" in data:
                    self.best_action = tuple(data["best_action"])
            logger.info(f"权重已加载: {path}")
        except Exception as e:
            logger.warning(f"加载权重失败: {e}, 使用默认权重")
    
    def _action_key(self, action: Tuple[float, int]) -> str:
        """生成action键名"""
        return f"{action[0]:.1f}_{action[1]}"
    
    def update_bandit(self, action: Tuple[float, int], success: bool):
        """
        更新Thompson Sampling Bandit状态
        
        Args:
            action: (min_score, topk)
            success: 是否成功（例如次日涨停）
        """
        key = self._action_key(action)
        if key not in self.bandit_state:
            self.bandit_state[key] = [1.0, 1.0]
        
        alpha, beta = self.bandit_state[key]
        if success:
            alpha += 1
        else:
            beta += 1
        
        self.bandit_state[key] = [alpha, beta]
    
    def sample_best_action(self) -> Tuple[float, int]:
        """
        使用Thompson Sampling推荐最佳阈值
        
        Returns:
            (min_score, topk)
        """
        samples = []
        for action in self.actions:
            key = self._action_key(action)
            alpha, beta = self.bandit_state.get(key, [1.0, 1.0])
            # 从 Beta分布采样
            sample = np.random.beta(alpha, beta)
            samples.append((sample, action))
        
        # 选择采样值最高的action
        samples.sort(reverse=True, key=lambda x: x[0])
        self.best_action = samples[0][1]
        
        logger.info(f"Thompson Sampling推荐: min_score={self.best_action[0]}, topk={self.best_action[1]}")
        
        return self.best_action
    
    def get_bandit_recommendation(self) -> Dict[str, Any]:
        """
        获取当前Bandit推荐
        
        Returns:
            推荐结果字典
        """
        return {
            "min_score": self.best_action[0],
            "topk": self.best_action[1],
            "bandit_state": self.bandit_state,
            "iteration": self.iteration
        }


class RLDecisionAgent:
    """强化学习决策Agent - 综合模型+进化模块"""
    
    def __init__(
        self, 
        use_neural_network: bool = True,
        model_path: str = None,
        weights_path: str = "rl_weights.json"
    ):
        """
        初始化决策Agent
        
        Args:
            use_neural_network: 是否使用神经网络 (否则使用加权打分)
            model_path: 预训练模型路径
            weights_path: 权重文件路径
        """
        self.use_neural_network = use_neural_network
        
        # 神经网络模型
        if use_neural_network:
            self.model = RLDecisionNetwork()
            if model_path:
                try:
                    self.model.load_state_dict(torch.load(model_path))
                    logger.info(f"模型加载成功: {model_path}")
                except Exception as e:
                    logger.warning(f"模型加载失败: {e}, 使用随机初始化")
            self.model.eval()
        
        # 自我进化模块
        self.evolution = SelfEvolutionModule()
        self.evolution.load_weights(weights_path)
        
        logger.info(f"RLDecisionAgent初始化完成 (神经网络: {use_neural_network})")
    
    def predict_score(self, features: StockFeatures) -> Tuple[float, Dict[str, Any]]:
        """
        预测股票得分
        
        Args:
            features: 股票特征
            
        Returns:
            (得分, 详情)
        """
        if self.use_neural_network:
            # 使用神经网络
            with torch.no_grad():
                x = torch.tensor(features.to_vector()).unsqueeze(0)
                score = self.model(x).item()
            
            method = "神经网络"
        else:
            # 使用加权打分(增强版)
            weights = self.evolution.get_weights()
            score = (
                # 基础特征
                features.consecutive_days * 10 * weights["consecutive_days"] +
                features.seal_ratio * 100 * weights["seal_ratio"] +
                features.quality_score * weights["quality_score"] +
                features.is_leader * 100 * weights["is_leader"] +
                features.auction_change * 10 * weights["auction_change"] +
                features.auction_strength * weights["auction_strength"] +
                features.bid_ask_ratio * 20 * weights["bid_ask_ratio"] +
                features.large_ratio * 100 * weights["large_ratio"] +
                features.stability * weights["stability"] +
                # 分时特征
                (features.vwap_slope * 100 if np.isfinite(features.vwap_slope) else 0) * weights["vwap_slope"] +
                (abs(features.max_drawdown) * 100 if np.isfinite(features.max_drawdown) else 0) * weights["max_drawdown"] +
                (features.afternoon_strength * 100 if np.isfinite(features.afternoon_strength) else 0) * weights["afternoon_strength"] +
                # 板块特征
                features.sector_heat * 100 * weights["sector_heat"] +
                features.sector_count * 10 * weights["sector_count"] +
                # 首板加分
                features.is_first_board * 100 * weights["is_first_board"]
            )
            method = "加权打分(增强版)"
        
        details = {
            "method": method,
            "weights": self.evolution.get_weights() if not self.use_neural_network else None
        }
        
        return score, details
    
    def rank_stocks(self, auction_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对股票进行排序
        
        Args:
            auction_report: 竞价分析报告
            
        Returns:
            排序后的股票列表
        """
        logger.info("=" * 60)
        logger.info("🤖 启动强化学习智能体选股决策...")
        logger.info("=" * 60)
        
        ranked_stocks = []
        
        for stock_data in auction_report["stocks"]:
            # 构建特征(增强版 - 16维)
            yesterday = stock_data.get("yesterday_info", {})
            auction = stock_data.get("auction_info", {})
            
            features = StockFeatures(
                # 基础特征
                consecutive_days=yesterday.get("consecutive_days", 0),
                seal_ratio=yesterday.get("seal_ratio", 0),
                quality_score=yesterday.get("quality_score", 0),
                is_leader=1.0 if yesterday.get("is_leader", False) else 0.0,
                auction_change=auction.get("final_change", 0),
                auction_strength=auction.get("strength", 0),
                bid_ask_ratio=auction.get("bid_ask_ratio", 1.0),
                large_ratio=auction.get("large_ratio", 0),
                stability=auction.get("stability", 0),
                # 分时特征 (新增)
                vwap_slope=yesterday.get("vwap_slope_morning", 0),
                max_drawdown=yesterday.get("max_drawdown_morning", 0),
                afternoon_strength=yesterday.get("afternoon_strength", 0),
                # 板块特征 (新增)
                sector_heat=yesterday.get("sector_heat", 0),
                sector_count=yesterday.get("sector_count", 0),
                # 首板标识 (新增)
                is_first_board=1.0 if yesterday.get("is_first_board", False) else 0.0
            )
            
            # 预测得分
            score, details = self.predict_score(features)
            
            # 解释涨停原因（新增）
            reason_scores = self.explain_reasons(features)
            top_reasons = [name for name, s in reason_scores if s > 0][:3]  # 前3个原因
            
            ranked_stocks.append({
                "symbol": stock_data["symbol"],
                "name": stock_data["name"],
                "rl_score": score,
                "details": details,
                "reasons": top_reasons,  # 新增: 涨停原因
                "reason_scores": reason_scores[:5],  # 新增: 前5个原因详细得分
                "features": features.__dict__,
                "yesterday_info": stock_data["yesterday_info"],
                "auction_info": stock_data["auction_info"]
            })
            
            logger.info(
                f"  {stock_data['symbol']} {stock_data['name']}: "
                f"RL得分 {score:.2f} ({details['method']})"
            )
        
        # 按RL得分排序
        ranked_stocks.sort(key=lambda x: x["rl_score"], reverse=True)
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 智能体决策排序结果 (Top 10)")
        logger.info("=" * 60)
        for i, stock in enumerate(ranked_stocks[:10], 1):
            reasons_str = ", ".join(stock.get("reasons", [])[:3]) or "无"
            logger.info(
                f"{i}. {stock['symbol']} {stock['name']}: "
                f"RL得分 {stock['rl_score']:.2f}, "
                f"连板 {stock['yesterday_info']['consecutive_days']}天, "
                f"竞价强度 {stock['auction_info']['strength']:.1f}\n"
                f"   → 涨停原因: {reasons_str}"
            )
        logger.info("=" * 60)
        
        return ranked_stocks
    
    def select_top_stocks(
        self, 
        ranked_stocks: List[Dict[str, Any]], 
        top_n: int = 5,
        min_score: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        选择最优股票
        
        Args:
            ranked_stocks: 排序后的股票
            top_n: 选择前N只
            min_score: 最低得分门槛
            
        Returns:
            筛选后的股票
        """
        selected = []
        
        for stock in ranked_stocks:
            if stock["rl_score"] >= min_score and len(selected) < top_n:
                selected.append(stock)
        
        logger.info(f"\n✅ 最终选中 {len(selected)} 只股票 (要求Top{top_n}, 得分>={min_score})")
        
        return selected
    
    def explain_reasons(self, features: StockFeatures) -> List[Tuple[str, float]]:
        """
        解释涨停原因（8大维度）
        
        Args:
            features: 股票特征
            
        Returns:
            [(原因名称, 得分)] 排序列表
        """
        reason_rules = [
            ("强竞价", lambda: features.vwap_slope >= 0.03),
            ("上午抗回撤", lambda: features.max_drawdown >= -0.02),
            ("午后延续性", lambda: features.afternoon_strength >= 0.01),
            ("题材热度高", lambda: features.sector_heat >= 0.7),
            ("量能放大", lambda: (features.bid_ask_ratio / 5) * (features.auction_strength / 100) >= 0.4),
            ("封板迅速", lambda: features.consecutive_days >= 1),
            ("封单强度高", lambda: features.seal_ratio >= 0.08),
            ("龙头地位", lambda: features.is_leader >= 0.5),
        ]
        
        scores = []
        for name, rule_fn in reason_rules:
            try:
                match = rule_fn()
                score = 1.0 if match else 0.0
            except Exception:
                score = 0.0
            scores.append((name, score))
        
        # 加权排序（关键因素权重更高）
        weights = {
            "强竞价": 1.2,
            "上午抗回撤": 1.0,
            "午后延续性": 1.0,
            "题材热度高": 1.2,
            "量能放大": 1.0,
            "封板迅速": 1.1,
            "封单强度高": 1.0,
            "龙头地位": 1.1,
        }
        
        weighted_scores = [
            (name, score * weights.get(name, 1.0))
            for name, score in scores
        ]
        
        return sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    
    def get_recommended_thresholds(self) -> Dict[str, Any]:
        """
        获取Thompson Sampling推荐的阈值
        
        Returns:
            {
                "min_score": float,
                "topk": int,
                "bandit_state": dict
            }
        """
        return self.evolution.get_bandit_recommendation()
    
    def sample_thresholds(self) -> Tuple[float, int]:
        """
        使用Thompson Sampling采样最佳阈值
        
        Returns:
            (min_score, topk)
        """
        return self.evolution.sample_best_action()
    
    def update_bandit_feedback(self, action: Tuple[float, int], success: bool):
        """
        更新Bandit反馈
        
        Args:
            action: (min_score, topk)
            success: 是否成功
        """
        self.evolution.update_bandit(action, success)
    
    def update_performance(self, symbol: str, predicted_score: float, actual_return: float):
        """
        更新历史表现 - 用于自我进化
        
        Args:
            symbol: 股票代码
            predicted_score: 预测得分
            actual_return: 实际收益率
        """
        self.evolution.update_weights(symbol, predicted_score, actual_return)
        logger.info(f"更新表现记录: {symbol}, 预测 {predicted_score:.2f}, 实际 {actual_return:.2%}")


if __name__ == "__main__":
    # 测试用例
    import json
    
    # 模拟竞价报告
    mock_report = {
        "date": "2025-01-15",
        "stocks": [
            {
                "symbol": "000001",
                "name": "平安银行",
                "yesterday_info": {
                    "consecutive_days": 2,
                    "seal_ratio": 0.15,
                    "is_leader": True,
                    "quality_score": 85
                },
                "auction_info": {
                    "final_price": 10.5,
                    "final_change": 5.2,
                    "strength": 78.5,
                    "stability": 85.0,
                    "bid_ask_ratio": 2.3,
                    "large_ratio": 0.4
                }
            },
            {
                "symbol": "300750",
                "name": "宁德时代",
                "yesterday_info": {
                    "consecutive_days": 1,
                    "seal_ratio": 0.08,
                    "is_leader": True,
                    "quality_score": 92
                },
                "auction_info": {
                    "final_price": 200.5,
                    "final_change": 7.8,
                    "strength": 88.2,
                    "stability": 90.0,
                    "bid_ask_ratio": 3.5,
                    "large_ratio": 0.6
                }
            }
        ]
    }
    
    # 创建决策Agent
    agent = RLDecisionAgent(use_neural_network=False)
    
    # 排序
    ranked = agent.rank_stocks(mock_report)
    
    # 选择
    selected = agent.select_top_stocks(ranked, top_n=2, min_score=70)
    
    print("\n最终选中股票:")
    for stock in selected:
        print(f"  {stock['symbol']} {stock['name']}: {stock['rl_score']:.2f}分")
