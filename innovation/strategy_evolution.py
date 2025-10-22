"""
AI策略进化系统
使用遗传算法和强化学习自动优化策略参数
支持策略组合优化和自适应进化
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import random
import copy

logger = logging.getLogger(__name__)


# ============================================================================
# 进化算法类型
# ============================================================================

class EvolutionType(Enum):
    """进化算法类型"""
    GENETIC = "genetic"           # 遗传算法
    PARTICLE_SWARM = "pso"        # 粒子群优化
    DIFFERENTIAL = "differential" # 差分进化
    REINFORCEMENT = "rl"          # 强化学习


@dataclass
class Individual:
    """个体（策略参数集）"""
    params: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, params={self.params})"


# ============================================================================
# 遗传算法策略优化器
# ============================================================================

class GeneticStrategyOptimizer:
    """遗传算法策略优化器"""
    
    def __init__(self,
                 param_space: Dict[str, Tuple[Any, Any]],
                 fitness_func: Callable,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_rate: float = 0.1):
        """
        初始化遗传算法优化器
        
        Args:
            param_space: 参数空间 {param_name: (min, max)}
            fitness_func: 适应度函数（越大越好）
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_rate: 精英保留比例
        """
        self.param_space = param_space
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_rate = elite_rate
        
        self.population = []
        self.best_individual = None
        self.history = []
        
        logger.info(f"遗传算法初始化: 种群={population_size}, 代数={generations}")
    
    def _random_params(self) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        for param_name, (min_val, max_val) in self.param_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = random.randint(min_val, max_val)
            else:
                params[param_name] = random.uniform(float(min_val), float(max_val))
        return params
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            params = self._random_params()
            individual = Individual(params=params, generation=0)
            self.population.append(individual)
        logger.info(f"初始化种群: {self.population_size} 个个体")
    
    def _evaluate_fitness(self, individual: Individual) -> float:
        """评估个体适应度"""
        try:
            fitness = self.fitness_func(individual.params)
            return fitness
        except Exception as e:
            logger.error(f"适应度计算失败: {e}")
            return -np.inf
    
    def _evaluate_population(self):
        """评估整个种群"""
        for individual in self.population:
            individual.fitness = self._evaluate_fitness(individual)
        
        # 排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # 更新最佳个体
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
    
    def _selection(self) -> List[Individual]:
        """选择操作（锦标赛选择）"""
        selected = []
        elite_count = int(self.population_size * self.elite_rate)
        
        # 精英保留
        selected.extend(copy.deepcopy(self.population[:elite_count]))
        
        # 锦标赛选择
        while len(selected) < self.population_size:
            # 随机选3个，取最好的
            tournament = random.sample(self.population, min(3, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))
        
        return selected
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1_params = {}
        child2_params = {}
        
        for param_name in self.param_space.keys():
            # 单点交叉
            if random.random() < 0.5:
                child1_params[param_name] = parent1.params[param_name]
                child2_params[param_name] = parent2.params[param_name]
            else:
                child1_params[param_name] = parent2.params[param_name]
                child2_params[param_name] = parent1.params[param_name]
        
        child1 = Individual(params=child1_params)
        child2 = Individual(params=child2_params)
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """变异操作"""
        mutated_params = copy.deepcopy(individual.params)
        
        for param_name, (min_val, max_val) in self.param_space.items():
            if random.random() < self.mutation_rate:
                # 高斯变异
                if isinstance(min_val, int) and isinstance(max_val, int):
                    current = mutated_params[param_name]
                    mutation = int(np.random.normal(0, (max_val - min_val) * 0.1))
                    mutated_params[param_name] = np.clip(current + mutation, min_val, max_val)
                else:
                    current = mutated_params[param_name]
                    mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                    mutated_params[param_name] = np.clip(current + mutation, min_val, max_val)
        
        return Individual(params=mutated_params)
    
    def evolve(self) -> Individual:
        """
        执行进化
        
        Returns:
            最佳个体
        """
        # 初始化
        self._initialize_population()
        self._evaluate_population()
        
        logger.info(f"初始最佳适应度: {self.population[0].fitness:.4f}")
        
        # 迭代进化
        for generation in range(self.generations):
            # 选择
            selected = self._selection()
            
            # 交叉和变异
            next_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]
                    child1, child2 = self._crossover(parent1, parent2)
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    child1.generation = generation + 1
                    child2.generation = generation + 1
                    next_population.extend([child1, child2])
                else:
                    next_population.append(self._mutate(selected[i]))
            
            self.population = next_population[:self.population_size]
            
            # 评估
            self._evaluate_population()
            
            # 记录历史
            self.history.append({
                'generation': generation + 1,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'best_params': self.population[0].params
            })
            
            if (generation + 1) % 10 == 0:
                logger.info(f"代数 {generation + 1}/{self.generations}: "
                          f"最佳={self.population[0].fitness:.4f}, "
                          f"平均={self.history[-1]['avg_fitness']:.4f}")
        
        logger.info(f"进化完成! 最佳适应度: {self.best_individual.fitness:.4f}")
        logger.info(f"最佳参数: {self.best_individual.params}")
        
        return self.best_individual
    
    def get_evolution_history(self) -> pd.DataFrame:
        """获取进化历史"""
        return pd.DataFrame(self.history)


# ============================================================================
# 强化学习策略优化器
# ============================================================================

class RLStrategyOptimizer:
    """强化学习策略优化器（基于Q-Learning）"""
    
    def __init__(self,
                 state_space: List[str],
                 action_space: Dict[str, List[Any]],
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 episodes: int = 1000):
        """
        初始化RL优化器
        
        Args:
            state_space: 状态空间（特征列表）
            action_space: 动作空间 {param_name: [possible_values]}
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            episodes: 训练轮数
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        
        # Q表
        self.q_table = {}
        
        # 历史
        self.history = []
        
        logger.info(f"强化学习初始化: 轮数={episodes}, ε={epsilon}")
    
    def _discretize_state(self, state: Dict[str, float]) -> str:
        """离散化状态"""
        # 简化：将连续状态离散化为字符串
        state_str = "_".join([f"{k}:{v:.2f}" for k, v in sorted(state.items())])
        return state_str
    
    def _get_action(self, state_str: str) -> Dict[str, Any]:
        """选择动作（ε-greedy策略）"""
        if random.random() < self.epsilon:
            # 探索：随机选择
            action = {param: random.choice(values) for param, values in self.action_space.items()}
        else:
            # 利用：选择Q值最大的动作
            if state_str not in self.q_table:
                self.q_table[state_str] = {}
            
            if not self.q_table[state_str]:
                action = {param: random.choice(values) for param, values in self.action_space.items()}
            else:
                action_str = max(self.q_table[state_str], key=self.q_table[state_str].get)
                action = eval(action_str)  # 简化处理
        
        return action
    
    def _update_q_value(self, state_str: str, action_str: str, reward: float, next_state_str: str):
        """更新Q值"""
        if state_str not in self.q_table:
            self.q_table[state_str] = {}
        
        current_q = self.q_table[state_str].get(action_str, 0.0)
        
        # 下一状态的最大Q值
        if next_state_str in self.q_table and self.q_table[next_state_str]:
            max_next_q = max(self.q_table[next_state_str].values())
        else:
            max_next_q = 0.0
        
        # Q-Learning更新
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_str][action_str] = new_q
    
    def train(self, env_step_func: Callable) -> Dict[str, Any]:
        """
        训练RL策略
        
        Args:
            env_step_func: 环境步进函数，接收(state, action)，返回(next_state, reward, done)
            
        Returns:
            最佳策略参数
        """
        total_rewards = []
        
        for episode in range(self.episodes):
            # 初始状态（简化）
            state = {feature: 0.0 for feature in self.state_space}
            state_str = self._discretize_state(state)
            
            episode_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < 100:
                # 选择动作
                action = self._get_action(state_str)
                action_str = str(action)
                
                # 环境步进
                next_state, reward, done = env_step_func(state, action)
                next_state_str = self._discretize_state(next_state)
                
                # 更新Q值
                self._update_q_value(state_str, action_str, reward, next_state_str)
                
                # 更新状态
                state = next_state
                state_str = next_state_str
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                logger.info(f"Episode {episode + 1}/{self.episodes}: 平均奖励={avg_reward:.4f}")
        
        # 提取最佳策略
        best_action = self._extract_best_policy()
        
        logger.info(f"训练完成! 最佳策略: {best_action}")
        return best_action
    
    def _extract_best_policy(self) -> Dict[str, Any]:
        """提取最佳策略"""
        # 简化：从Q表中选择平均Q值最高的动作
        action_q_values = {}
        
        for state_actions in self.q_table.values():
            for action_str, q_value in state_actions.items():
                if action_str not in action_q_values:
                    action_q_values[action_str] = []
                action_q_values[action_str].append(q_value)
        
        if not action_q_values:
            return {param: random.choice(values) for param, values in self.action_space.items()}
        
        best_action_str = max(action_q_values, key=lambda k: np.mean(action_q_values[k]))
        return eval(best_action_str)


# ============================================================================
# 策略组合优化器
# ============================================================================

class PortfolioStrategyOptimizer:
    """策略组合优化器"""
    
    def __init__(self, strategies: List[Dict[str, Any]]):
        """
        初始化组合优化器
        
        Args:
            strategies: 策略列表，每个策略包含name、returns等信息
        """
        self.strategies = strategies
        self.optimal_weights = None
        
        logger.info(f"策略组合优化器初始化: {len(strategies)} 个策略")
    
    def optimize_weights(self, 
                        returns_df: pd.DataFrame,
                        method: str = "sharpe") -> Dict[str, float]:
        """
        优化策略权重
        
        Args:
            returns_df: 策略收益率DataFrame，每列一个策略
            method: 优化目标 (sharpe, min_variance, max_return)
            
        Returns:
            最优权重字典
        """
        n_strategies = len(returns_df.columns)
        
        if method == "sharpe":
            # 最大化Sharpe比率
            optimal_weights = self._maximize_sharpe(returns_df)
        elif method == "min_variance":
            # 最小化方差
            optimal_weights = self._minimize_variance(returns_df)
        elif method == "max_return":
            # 最大化收益
            optimal_weights = self._maximize_return(returns_df)
        else:
            # 等权重
            optimal_weights = np.ones(n_strategies) / n_strategies
        
        # 转换为字典
        self.optimal_weights = {
            strategy: weight 
            for strategy, weight in zip(returns_df.columns, optimal_weights)
        }
        
        logger.info(f"优化完成: {self.optimal_weights}")
        return self.optimal_weights
    
    def _maximize_sharpe(self, returns_df: pd.DataFrame) -> np.ndarray:
        """最大化Sharpe比率"""
        try:
            from scipy.optimize import minimize
            
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            n = len(mean_returns)
            
            def neg_sharpe(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -portfolio_return / portfolio_std if portfolio_std > 0 else 0
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((0, 1) for _ in range(n))
            initial_weights = np.ones(n) / n
            
            result = minimize(neg_sharpe, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else initial_weights
            
        except ImportError:
            logger.warning("scipy未安装，使用等权重")
            return np.ones(len(returns_df.columns)) / len(returns_df.columns)
    
    def _minimize_variance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """最小化方差"""
        try:
            from scipy.optimize import minimize
            
            cov_matrix = returns_df.cov()
            n = len(returns_df.columns)
            
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((0, 1) for _ in range(n))
            initial_weights = np.ones(n) / n
            
            result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            return result.x if result.success else initial_weights
            
        except ImportError:
            return np.ones(len(returns_df.columns)) / len(returns_df.columns)
    
    def _maximize_return(self, returns_df: pd.DataFrame) -> np.ndarray:
        """最大化收益（简化：按历史收益率加权）"""
        mean_returns = returns_df.mean()
        weights = mean_returns / mean_returns.sum()
        return weights.values


# ============================================================================
# 使用示例
# ============================================================================

def example_strategy_evolution():
    """策略进化示例"""
    print("=== AI策略进化系统示例 ===\n")
    
    # 1. 遗传算法优化
    print("1. 遗传算法策略优化")
    
    param_space = {
        'window': (5, 50),
        'threshold': (0.01, 0.1),
        'stop_loss': (0.02, 0.1)
    }
    
    def fitness_function(params):
        """模拟适应度函数（Sharpe比率）"""
        # 实际应运行回测计算真实Sharpe
        sharpe = np.random.random() * 2
        # 加入参数合理性惩罚
        if params['window'] < 10:
            sharpe *= 0.8
        return sharpe
    
    ga_optimizer = GeneticStrategyOptimizer(
        param_space=param_space,
        fitness_func=fitness_function,
        population_size=20,
        generations=30,
        mutation_rate=0.1
    )
    
    best_individual = ga_optimizer.evolve()
    print(f"  最佳参数: {best_individual.params}")
    print(f"  最佳适应度: {best_individual.fitness:.4f}")
    
    # 2. 策略组合优化
    print("\n2. 策略组合优化")
    
    # 模拟策略收益率
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns_df = pd.DataFrame({
        'momentum': np.random.randn(252) * 0.02,
        'mean_reversion': np.random.randn(252) * 0.015,
        'trend_following': np.random.randn(252) * 0.018
    }, index=dates)
    
    portfolio_optimizer = PortfolioStrategyOptimizer(
        strategies=[{'name': col} for col in returns_df.columns]
    )
    
    optimal_weights = portfolio_optimizer.optimize_weights(returns_df, method="sharpe")
    print(f"  最优权重:")
    for strategy, weight in optimal_weights.items():
        print(f"    {strategy}: {weight:.2%}")
    
    # 计算组合表现
    portfolio_returns = (returns_df * pd.Series(optimal_weights)).sum(axis=1)
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    print(f"  组合Sharpe: {sharpe:.2f}")
    
    print("\n策略进化系统演示完成!")


if __name__ == "__main__":
    example_strategy_evolution()
