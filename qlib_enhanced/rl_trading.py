"""
强化学习交易策略模块
支持DQN、PPO、A2C等算法进行股票交易策略训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import pickle

logger = logging.getLogger(__name__)


# ============================================================================
# 交易动作定义
# ============================================================================

class TradingAction(Enum):
    """交易动作"""
    HOLD = 0      # 持有
    BUY = 1       # 买入
    SELL = 2      # 卖出


@dataclass
class TradeExecution:
    """交易执行记录"""
    timestamp: datetime
    symbol: str
    action: TradingAction
    quantity: int
    price: float
    value: float
    commission: float


# ============================================================================
# 交易环境
# ============================================================================

class TradingEnvironment:
    """
    强化学习交易环境
    模拟真实股票市场交易
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 100000.0,
                 commission_rate: float = 0.001,
                 max_position: int = 1000):
        """
        初始化交易环境
        
        Args:
            data: 历史价格数据 (columns: open, high, low, close, volume)
            initial_balance: 初始资金
            commission_rate: 手续费率
            max_position: 最大持仓数量
        """
        self.data = data
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_position = max_position
        
        # 环境状态
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 当前持仓
        self.total_value = initial_balance
        self.trades = []
        
        # 性能指标
        self.max_value = initial_balance
        self.max_drawdown = 0.0
        
        logger.info(f"交易环境初始化: 初始资金={initial_balance}, 数据长度={len(data)}")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_value = self.initial_balance
        self.trades = []
        self.max_value = self.initial_balance
        self.max_drawdown = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态
        
        Returns:
            状态向量: [价格特征, 持仓信息, 账户信息]
        """
        if self.current_step >= len(self.data):
            return np.zeros(20)  # 默认状态维度
        
        # 价格特征 (最近5天)
        lookback = 5
        start_idx = max(0, self.current_step - lookback + 1)
        recent_data = self.data.iloc[start_idx:self.current_step + 1]
        
        price_features = []
        for col in ['close', 'volume']:
            if col in recent_data.columns:
                values = recent_data[col].values
                # 归一化
                if len(values) > 0:
                    normalized = (values - values.mean()) / (values.std() + 1e-8)
                    # 填充到固定长度
                    if len(normalized) < lookback:
                        normalized = np.pad(normalized, (lookback - len(normalized), 0))
                    price_features.extend(normalized[-lookback:])
        
        # 持仓信息
        current_price = self.data.iloc[self.current_step]['close']
        position_ratio = self.position / self.max_position if self.max_position > 0 else 0
        position_value = self.position * current_price
        position_features = [
            position_ratio,
            position_value / self.initial_balance if self.initial_balance > 0 else 0
        ]
        
        # 账户信息
        total_value = self.balance + position_value
        account_features = [
            self.balance / self.initial_balance if self.initial_balance > 0 else 0,
            total_value / self.initial_balance if self.initial_balance > 0 else 0,
            (total_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        ]
        
        # 组合状态
        state = np.array(price_features + position_features + account_features, dtype=np.float32)
        
        # 确保固定维度
        if len(state) < 20:
            state = np.pad(state, (0, 20 - len(state)))
        elif len(state) > 20:
            state = state[:20]
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 0=持有, 1=买入, 2=卖出
        
        Returns:
            next_state, reward, done, info
        """
        current_price = self.data.iloc[self.current_step]['close']
        trading_action = TradingAction(action)
        
        # 执行交易
        reward = 0.0
        trade_info = None
        
        if trading_action == TradingAction.BUY and self.balance > 0:
            # 买入
            max_buy = int(self.balance / (current_price * (1 + self.commission_rate)))
            buy_quantity = min(max_buy, self.max_position - self.position)
            
            if buy_quantity > 0:
                cost = buy_quantity * current_price * (1 + self.commission_rate)
                self.balance -= cost
                self.position += buy_quantity
                
                trade_info = TradeExecution(
                    timestamp=self.data.index[self.current_step],
                    symbol='stock',
                    action=TradingAction.BUY,
                    quantity=buy_quantity,
                    price=current_price,
                    value=cost,
                    commission=buy_quantity * current_price * self.commission_rate
                )
                self.trades.append(trade_info)
        
        elif trading_action == TradingAction.SELL and self.position > 0:
            # 卖出全部持仓
            sell_value = self.position * current_price * (1 - self.commission_rate)
            self.balance += sell_value
            
            trade_info = TradeExecution(
                timestamp=self.data.index[self.current_step],
                symbol='stock',
                action=TradingAction.SELL,
                quantity=self.position,
                price=current_price,
                value=sell_value,
                commission=self.position * current_price * self.commission_rate
            )
            self.trades.append(trade_info)
            self.position = 0
        
        # 移动到下一步
        self.current_step += 1
        
        # 计算奖励
        if self.current_step < len(self.data):
            next_price = self.data.iloc[self.current_step]['close']
            position_value = self.position * next_price
            new_total_value = self.balance + position_value
            
            # 奖励 = 总资产变化率
            reward = (new_total_value - self.total_value) / self.total_value if self.total_value > 0 else 0
            
            self.total_value = new_total_value
            
            # 更新最大回撤
            if self.total_value > self.max_value:
                self.max_value = self.total_value
            drawdown = (self.max_value - self.total_value) / self.max_value if self.max_value > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # 判断是否结束
        done = self.current_step >= len(self.data) - 1
        
        # 下一个状态
        next_state = self._get_state() if not done else np.zeros(20)
        
        # 信息
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_value': self.total_value,
            'max_drawdown': self.max_drawdown,
            'trade': trade_info
        }
        
        return next_state, reward, done, info
    
    def get_performance(self) -> Dict[str, float]:
        """获取性能指标"""
        total_return = (self.total_value - self.initial_balance) / self.initial_balance
        num_trades = len(self.trades)
        
        # 计算夏普比率
        if len(self.trades) > 1:
            returns = []
            for i in range(1, len(self.trades)):
                returns.append((self.trades[i].value - self.trades[i-1].value) / self.trades[i-1].value)
            
            if len(returns) > 0:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        return {
            'total_return': total_return,
            'total_value': self.total_value,
            'max_drawdown': self.max_drawdown,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe
        }


# ============================================================================
# DQN智能体
# ============================================================================

class DQNAgent:
    """
    Deep Q-Network (DQN) 交易智能体
    """
    
    def __init__(self,
                 state_dim: int = 20,
                 action_dim: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q网络 (简化版，实际应使用神经网络)
        self.q_table = {}
        
        # 经验回放
        self.memory = []
        self.memory_size = 10000
        
        logger.info(f"DQN智能体初始化: state_dim={state_dim}, action_dim={action_dim}")
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """将状态转换为字典键"""
        # 离散化状态
        discretized = (state * 10).astype(int)
        return str(discretized.tolist())
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否训练模式
        
        Returns:
            动作索引
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(0, self.action_dim)
        else:
            # 利用：选择最优动作
            state_key = self._state_to_key(state)
            if state_key not in self.q_table:
                return np.random.randint(0, self.action_dim)
            
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def train(self, batch_size: int = 32):
        """训练Q网络"""
        if len(self.memory) < batch_size:
            return 0.0
        
        # 随机采样
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        total_loss = 0.0
        
        for state, action, reward, next_state, done in batch:
            state_key = self._state_to_key(state)
            next_state_key = self._state_to_key(next_state)
            
            # 初始化Q值
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_dim)
            
            # Q-learning更新
            current_q = self.q_table[state_key][action]
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_table[next_state_key])
            
            # 更新Q值
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
            
            total_loss += abs(target_q - current_q)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size
    
    def save(self, filepath: str):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
        logger.info(f"模型已保存: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
        logger.info(f"模型已加载: {filepath}")


# ============================================================================
# 训练器
# ============================================================================

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self,
                 env: TradingEnvironment,
                 agent: DQNAgent):
        """
        初始化训练器
        
        Args:
            env: 交易环境
            agent: RL智能体
        """
        self.env = env
        self.agent = agent
        self.training_history = []
    
    def train(self, num_episodes: int = 100, batch_size: int = 32) -> List[Dict]:
        """
        训练智能体
        
        Args:
            num_episodes: 训练轮数
            batch_size: 批次大小
        
        Returns:
            训练历史
        """
        logger.info(f"开始训练: {num_episodes} 轮")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            steps = 0
            
            while not done:
                # 选择动作
                action = self.agent.choose_action(state, training=True)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # 训练
                loss = self.agent.train(batch_size)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            # 记录训练历史
            performance = self.env.get_performance()
            history_entry = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': steps,
                'epsilon': self.agent.epsilon,
                **performance
            }
            self.training_history.append(history_entry)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{num_episodes}: "
                          f"Return={performance['total_return']:.2%}, "
                          f"Value={performance['total_value']:.2f}, "
                          f"Epsilon={self.agent.epsilon:.3f}")
        
        logger.info("训练完成！")
        return self.training_history
    
    def backtest(self) -> Dict[str, Any]:
        """回测智能体"""
        logger.info("开始回测...")
        
        state = self.env.reset()
        done = False
        actions_taken = []
        
        while not done:
            action = self.agent.choose_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            actions_taken.append({
                'step': self.env.current_step - 1,
                'action': TradingAction(action).name,
                'balance': info['balance'],
                'position': info['position'],
                'total_value': info['total_value']
            })
            
            state = next_state
        
        performance = self.env.get_performance()
        
        logger.info(f"回测完成: 总收益={performance['total_return']:.2%}")
        
        return {
            'performance': performance,
            'actions': actions_taken,
            'trades': self.env.trades
        }


# ============================================================================
# 使用示例
# ============================================================================

def create_sample_data(days: int = 252) -> pd.DataFrame:
    """创建模拟数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # 模拟价格走势
    prices = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.02))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(days) * 0.02)),
        'low': prices * (1 - np.abs(np.random.randn(days) * 0.02)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    return data


def main():
    """示例：训练DQN交易策略"""
    print("=" * 80)
    print("强化学习交易策略 - DQN训练示例")
    print("=" * 80)
    
    # 1. 创建环境
    print("\n📊 创建交易环境...")
    data = create_sample_data(days=252)
    env = TradingEnvironment(data, initial_balance=100000.0)
    
    # 2. 创建智能体
    print("🤖 创建DQN智能体...")
    agent = DQNAgent(state_dim=20, action_dim=3)
    
    # 3. 训练
    print("\n🎯 开始训练...")
    trainer = RLTrainer(env, agent)
    history = trainer.train(num_episodes=50, batch_size=32)
    
    # 4. 回测
    print("\n📈 回测策略...")
    backtest_results = trainer.backtest()
    
    # 5. 显示结果
    print("\n" + "=" * 80)
    print("📊 训练结果")
    print("=" * 80)
    
    final_performance = history[-1]
    print(f"总收益率: {final_performance['total_return']:.2%}")
    print(f"最终资产: ¥{final_performance['total_value']:,.2f}")
    print(f"最大回撤: {final_performance['max_drawdown']:.2%}")
    print(f"夏普比率: {final_performance['sharpe_ratio']:.2f}")
    print(f"交易次数: {final_performance['num_trades']}")
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    from qilin_stack.app.core.logging_setup import setup_logging
    setup_logging()
    main()
