"""
Pytest fixtures和配置
提供测试所需的通用fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import sys
import tempfile
import shutil

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# 允许引用上层仓库根目录下的 app/* 模块（如 app.pool）
sys.path.insert(0, str(project_root.parent))

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_ohlcv_data():
    """生成示例OHLCV数据"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    return pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.3,
        'high': close_prices + np.abs(np.random.randn(100) * 0.5),
        'low': close_prices - np.abs(np.random.randn(100) * 0.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

@pytest.fixture
def sample_tick_data():
    """生成示例Tick数据"""
    n_ticks = 1000
    base_price = 100
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01 09:30:00', periods=n_ticks, freq='S'),
        'price': base_price + np.cumsum(np.random.randn(n_ticks) * 0.01),
        'volume': np.random.randint(100, 1000, n_ticks),
        'bid_price': base_price + np.cumsum(np.random.randn(n_ticks) * 0.01) - 0.01,
        'ask_price': base_price + np.cumsum(np.random.randn(n_ticks) * 0.01) + 0.01,
        'bid_volume': np.random.randint(1000, 10000, n_ticks),
        'ask_volume': np.random.randint(1000, 10000, n_ticks)
    })

@pytest.fixture
def sample_symbols():
    """生成示例股票代码列表"""
    return ['000001', '000002', '000858', '002142', '300750', '600519']

# 追加通用fixtures
@pytest.fixture
def sample_date():
    return '2024-06-30'

@pytest.fixture
def date_range():
    return {'start_date': '2024-01-01', 'end_date': '2024-06-30'}

@pytest.fixture
def sample_market_data():
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    return pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(200)),
        'volume': np.random.randint(1_000_000, 2_000_000, 200)
    }, index=dates)

@pytest.fixture
def mock_market_data():
    """模拟市场数据"""
    return {
        'market_mood_score': 65,
        'sector_heat': {
            'technology': 8.5,
            'finance': 6.2,
            'consumer': 7.3
        },
        'index_data': {
            'sh_index': {'change': 1.2, 'volume_ratio': 1.15},
            'sz_index': {'change': 1.5, 'volume_ratio': 1.25}
        },
        'turnover_rate': 2.5,
        'money_flow': {
            'main_inflow': 1500000000,
            'retail_inflow': -500000000
        }
    }

@pytest.fixture
def temp_directory():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config():
    """模拟配置"""
    return {
        'system': {
            'name': 'test_system',
            'mode': 'simulation',
            'timezone': 'Asia/Shanghai'
        },
        'trading': {
            'symbols': ['000001', '000002'],
            'max_positions': 5,
            'position_size': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.10
        },
        'risk': {
            'max_daily_loss': 0.02,
            'max_position_risk': 0.01,
            'max_correlation': 0.7
        },
        'data': {
            'sources': ['mock'],
            'cache': {
                'enabled': True,
                'ttl': 300
            }
        }
    }

@pytest.fixture
def mlflow_tracking_uri(temp_directory):
    """MLflow tracking URI"""
    mlflow_dir = temp_directory / 'mlruns'
    mlflow_dir.mkdir(exist_ok=True)
    return f"file://{mlflow_dir}"

@pytest.fixture
def sample_model_data():
    """生成模型训练数据"""
    n_samples = 1000
    n_features = 20
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1) > 0
    y = y.astype(int)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

@pytest.fixture
def sample_portfolio():
    """生成示例投资组合"""
    return {
        'cash': 100000,
        'positions': {
            '000001': {'quantity': 1000, 'avg_price': 50.0, 'current_price': 52.0},
            '000002': {'quantity': 500, 'avg_price': 100.0, 'current_price': 98.0}
        },
        'total_value': 152000,
        'pnl': 2000,
        'return': 0.02
    }

@pytest.fixture
def sample_orders():
    """生成示例订单"""
    return [
        {
            'order_id': 'order_001',
            'symbol': '000001',
            'side': 'buy',
            'quantity': 100,
            'price': 50.0,
            'status': 'filled',
            'timestamp': datetime.now()
        },
        {
            'order_id': 'order_002',
            'symbol': '000002',
            'side': 'sell',
            'quantity': 50,
            'price': 100.0,
            'status': 'filled',
            'timestamp': datetime.now()
        }
    ]

@pytest.fixture
def sample_trades():
    """生成示例成交记录"""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    
    trades = []
    for date in dates:
        trade = {
            'timestamp': date,
            'symbol': np.random.choice(['000001', '000002', '000858']),
            'side': np.random.choice(['buy', 'sell']),
            'quantity': np.random.randint(100, 1000),
            'price': np.random.uniform(50, 150),
            'commission': np.random.uniform(10, 100),
            'pnl': np.random.uniform(-500, 500)
        }
        trades.append(trade)
    
    return pd.DataFrame(trades)

@pytest.fixture
def redis_client(mock_config):
    """模拟Redis客户端"""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ex=None):
            self.data[key] = value
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
        
        def exists(self, key):
            return key in self.data
        
        def keys(self, pattern='*'):
            import re
            regex = re.compile(pattern.replace('*', '.*'))
            return [k for k in self.data.keys() if regex.match(k)]
        
        def flushdb(self):
            self.data.clear()
    
    return MockRedis()

@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个测试前重置随机种子"""
    np.random.seed(42)
    yield

@pytest.fixture
def mock_agent_responses():
    """模拟Agent响应"""
    return [
        {
            'agent_id': 'market_ecology',
            'score': 75.5,
            'confidence': 0.85,
            'details': {'market_strength': 'strong'}
        },
        {
            'agent_id': 'technical',
            'score': 68.2,
            'confidence': 0.78,
            'details': {'trend': 'bullish'}
        },
        {
            'agent_id': 'risk',
            'score': 82.0,
            'confidence': 0.92,
            'details': {'risk_level': 'moderate'}
        }
    ]

@pytest.fixture(scope="session")
def docker_services():
    """检查Docker服务是否可用"""
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', 'ps'], capture_output=True, text=True, check=False)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

@pytest.fixture
def benchmark_data():
    """基准数据用于性能测试"""
    return {
        'symbols': [f'00000{i}' for i in range(100)],
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'expected_processing_time': 10.0,  # 秒
        'max_memory_mb': 500
    }

# 测试配置钩子
def pytest_configure(config):
    """Pytest配置钩子"""
    config.addinivalue_line(
        "markers",
        "requires_mlflow: 测试需要MLflow服务运行"
    )
    config.addinivalue_line(
        "markers",
        "requires_redis: 测试需要Redis服务运行"
    )
    config.addinivalue_line(
        "markers",
        "requires_docker: 测试需要Docker运行"
    )

def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    # 为慢速测试添加标记
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # 为集成测试添加标记
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


