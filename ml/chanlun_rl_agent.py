"""强化学习策略优化 - P1-5

功能:
- Gym环境封装
- PPO/DQN训练框架
- 自适应参数调整
- 策略评估与回测

注意:
⚠️ 需要GPU加速训练
⚠️ 需要大量历史数据(5年+)
⚠️ 训练时间长(数周)

作者: Warp AI Assistant
日期: 2025-01
"""
import gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    logger.warning("stable-baselines3未安装, 部分功能不可用")


class ChanLunRLEnv(gym.Env):
    """缠论强化学习环境"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        stock_data: pd.DataFrame,
        initial_balance: float = 100000,
        commission_rate: float = 0.0003,
        max_steps: int = 252  # 一年交易日
    ):
        """初始化
        
        Args:
            stock_data: 股票历史数据 (含缠论特征)
            initial_balance: 初始资金
            commission_rate: 手续费率
            max_steps: 最大步数
        """
        super().__init__()
        
        self.stock_data = stock_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_steps = max_steps
        
        # 动作空间: 0=持有, 1=买入, 2=卖出
        self.action_space = gym.spaces.Discrete(3)
        
        # 观察空间: 30个特征
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32
        )
        
        # 状态
        self.current_step = 0
        self.position = 0  # 0=空仓, >0=持仓数量
        self.balance = initial_balance
        self.entry_price = 0
        
        logger.info(f"RL环境初始化: {len(stock_data)}条数据, 初始资金{initial_balance}")
    
    def reset(self):
        """重置环境"""
        self.current_step = 20  # 从第20条开始(需要历史数据)
        self.position = 0
        self.balance = self.initial_balance
        self.entry_price = 0
        
        return self._get_observation()
    
    def step(self, action):
        """执行一步
        
        Args:
            action: 0=持有, 1=买入, 2=卖出
        
        Returns:
            observation, reward, done, info
        """
        current_price = self.stock_data.iloc[self.current_step]['close']
        
        # 执行动作
        reward = 0
        if action == 1 and self.position == 0:  # 买入
            shares = int(self.balance / current_price * (1 - self.commission_rate))
            cost = shares * current_price * (1 + self.commission_rate)
            
            if cost <= self.balance:
                self.position = shares
                self.balance -= cost
                self.entry_price = current_price
                reward = -0.01  # 小惩罚(交易成本)
        
        elif action == 2 and self.position > 0:  # 卖出
            proceeds = self.position * current_price * (1 - self.commission_rate)
            self.balance += proceeds
            
            # 计算收益作为奖励
            profit = (current_price - self.entry_price) / self.entry_price
            reward = profit * 100  # 放大100倍
            
            self.position = 0
            self.entry_price = 0
        
        # 持仓奖励/惩罚
        if self.position > 0:
            unrealized_profit = (current_price - self.entry_price) / self.entry_price
            reward += unrealized_profit * 0.1  # 小幅奖励未实现收益
        
        # 下一步
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1 or self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_value': self.balance + self.position * current_price
        }
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态 (30个特征)"""
        if self.current_step >= len(self.stock_data):
            return np.zeros(30, dtype=np.float32)
        
        # 提取特征
        row = self.stock_data.iloc[self.current_step]
        
        # 价格特征 (5)
        features = [
            row['close'] / 100,  # 归一化
            row['high'] / row['low'] - 1,
            row['volume'] / 10000000,
            (row['close'] - row['open']) / row['open'],
            0  # 占位
        ]
        
        # 缠论特征 (10)
        chan_features = [
            row.get('is_buy_point', 0),
            row.get('is_sell_point', 0),
            row.get('bsp_type', 0) / 3,
            row.get('seg_direction', 0),
            row.get('in_chanpy_zs', 0),
            row.get('trend_type', 0),
            row.get('trend_strength', 0),
            0, 0, 0  # 占位
        ]
        features.extend(chan_features)
        
        # 技术指标 (10)
        if self.current_step >= 5:
            recent_closes = self.stock_data.iloc[self.current_step-5:self.current_step]['close'].values
            ma5 = recent_closes.mean()
            tech_features = [
                (row['close'] - ma5) / ma5,
                0, 0, 0, 0, 0, 0, 0, 0, 0  # 占位
            ]
        else:
            tech_features = [0] * 10
        features.extend(tech_features)
        
        # 持仓状态 (5)
        position_features = [
            1 if self.position > 0 else 0,
            (self.balance / self.initial_balance) - 1,
            (row['close'] - self.entry_price) / self.entry_price if self.entry_price > 0 else 0,
            0, 0
        ]
        features.extend(position_features)
        
        return np.array(features[:30], dtype=np.float32)
    
    def render(self, mode='human'):
        """可视化(可选)"""
        if mode == 'human':
            current_value = self.balance + self.position * self.stock_data.iloc[self.current_step]['close']
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}, Total: {current_value:.2f}")


class ChanLunRLTrainer:
    """缠论强化学习训练器"""
    
    def __init__(self, algorithm: str = 'PPO'):
        """初始化
        
        Args:
            algorithm: 'PPO' or 'DQN'
        """
        if not HAS_SB3:
            raise ImportError("请安装 stable-baselines3: pip install stable-baselines3")
        
        self.algorithm = algorithm
        self.model = None
        self.env = None
        
        logger.info(f"RL训练器初始化: {algorithm}")
    
    def prepare_training_data(self, stock_universe: list) -> pd.DataFrame:
        """准备训练数据
        
        Args:
            stock_universe: 股票列表
        
        Returns:
            合并的训练数据
        """
        # TODO: 加载真实股票数据
        # 这里用模拟数据
        logger.warning("使用模拟数据, 实际使用请加载真实数据")
        
        n = 1000
        data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=n, freq='D'),
            'open': 10 + np.random.randn(n).cumsum() * 0.1,
            'close': 10 + np.random.randn(n).cumsum() * 0.1,
            'high': 10.5 + np.random.randn(n).cumsum() * 0.1,
            'low': 9.5 + np.random.randn(n).cumsum() * 0.1,
            'volume': np.random.randint(1000000, 10000000, n),
            # 缠论特征
            'is_buy_point': np.random.choice([0, 1], n, p=[0.95, 0.05]),
            'is_sell_point': np.random.choice([0, 1], n, p=[0.95, 0.05]),
            'bsp_type': np.random.choice([0, 1, 2, 3], n, p=[0.90, 0.03, 0.04, 0.03]),
            'seg_direction': np.random.choice([-1, 1], n),
            'in_chanpy_zs': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'trend_type': np.random.choice([-1, 0, 1], n),
            'trend_strength': np.random.rand(n)
        })
        
        return data
    
    def train(
        self,
        training_data: pd.DataFrame,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None
    ):
        """训练模型
        
        Args:
            training_data: 训练数据
            total_timesteps: 总训练步数
            save_path: 模型保存路径
        """
        logger.info(f"开始训练: {total_timesteps} timesteps")
        
        # 创建环境
        env = ChanLunRLEnv(training_data)
        self.env = DummyVecEnv([lambda: env])
        
        # 创建模型
        if self.algorithm == 'PPO':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10
            )
        elif self.algorithm == 'DQN':
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=0.0001,
                buffer_size=50000,
                learning_starts=1000
            )
        else:
            raise ValueError(f"未知算法: {self.algorithm}")
        
        # 训练
        self.model.learn(total_timesteps=total_timesteps)
        
        # 保存模型
        if save_path:
            self.model.save(save_path)
            logger.info(f"模型已保存: {save_path}")
        
        return self.model
    
    def evaluate(self, test_data: pd.DataFrame, num_episodes: int = 10):
        """评估模型
        
        Args:
            test_data: 测试数据
            num_episodes: 测试回合数
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        env = ChanLunRLEnv(test_data)
        
        total_rewards = []
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, Final Value={info['total_value']:.2f}")
        
        avg_reward = np.mean(total_rewards)
        logger.info(f"平均奖励: {avg_reward:.2f}")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': np.std(total_rewards),
            'rewards': total_rewards
        }


if __name__ == '__main__':
    print("="*60)
    print("P1-5: 强化学习策略优化测试")
    print("="*60)
    
    if not HAS_SB3:
        print("\n⚠️  stable-baselines3未安装")
        print("   安装命令: pip install stable-baselines3")
        print("   需要PyTorch支持")
    else:
        print("\n✅ 依赖已安装")
        
        # 测试环境
        print("\n测试1: 创建RL环境")
        n = 100
        mock_data = pd.DataFrame({
            'close': 10 + np.random.randn(n).cumsum() * 0.1,
            'open': 10 + np.random.randn(n).cumsum() * 0.1,
            'high': 10.5 + np.random.randn(n).cumsum() * 0.1,
            'low': 9.5 + np.random.randn(n).cumsum() * 0.1,
            'volume': np.random.randint(1000000, 10000000, n),
            'is_buy_point': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'is_sell_point': 0,
            'bsp_type': 0,
            'seg_direction': 1,
            'in_chanpy_zs': 0,
            'trend_type': 0,
            'trend_strength': 0
        })
        
        env = ChanLunRLEnv(mock_data, max_steps=50)
        obs = env.reset()
        print(f"✅ 环境创建成功, 观察维度: {obs.shape}")
        
        # 测试几步
        print("\n测试2: 执行随机动作")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"   Step {i+1}: Action={action}, Reward={reward:.4f}, Balance={info['balance']:.2f}")
        
        print("\n✅ P1-5测试完成!")
        print("\n⚠️  完整训练需要:")
        print("   1. 大量历史数据 (5年+)")
        print("   2. GPU加速 (RTX 3090+)")
        print("   3. 长时间训练 (数周)")
