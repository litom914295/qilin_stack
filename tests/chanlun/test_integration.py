"""Week 1集成测试 - CZSC Handler"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from features.chanlun.czsc_features import CzscFeatureGenerator

def test_czsc_handler_mock():
    """测试CZSC Handler (Mock模式，无需Qlib数据)"""
    print("="*60)
    print("Week 1集成测试: CZSC Handler (Mock模式)")
    print("="*60)
    
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', periods=100)
    np.random.seed(42)
    
    # 模拟2只股票的数据
    stocks = ['SH600000', 'SH600036']
    all_data = []
    
    for stock in stocks:
        df = pd.DataFrame({
            'datetime': dates,
            'open': 10 + np.random.randn(100).cumsum() * 0.5,
            'close': 10 + np.random.randn(100).cumsum() * 0.5,
            'high': 10.5 + np.random.randn(100).cumsum() * 0.5,
            'low': 9.5 + np.random.randn(100).cumsum() * 0.5,
            'volume': np.random.randint(1000, 10000, 100),
            'symbol': stock
        })
        all_data.append(df)
    
    print(f"\n✅ 准备测试数据: {len(stocks)}只股票, 每只{len(dates)}天")
    
    # 测试特征生成
    generator = CzscFeatureGenerator()
    results = []
    
    for stock_df in all_data:
        result = generator.generate_features(stock_df)
        results.append(result)
        
        # 验证特征列
        chanlun_features = ['fx_mark', 'bi_direction', 'bi_position', 
                           'bi_power', 'in_zs', 'bars_since_fx']
        
        for feat in chanlun_features:
            assert feat in result.columns, f"缺少特征: {feat}"
        
        print(f"✅ {stock_df['symbol'].iloc[0]} 特征生成成功")
    
    # 统计结果
    all_results = pd.concat(results, ignore_index=True)
    
    print(f"\n📊 特征统计:")
    print(f"   总数据量: {len(all_results)}行")
    print(f"   特征列数: {len(chanlun_features)}")
    print(f"   分型数: {(all_results['fx_mark'] != 0).sum()}")
    print(f"   笔段数: {(all_results['bi_direction'] != 0).sum()}")
    
    # 验证数据质量
    for col in chanlun_features:
        null_count = all_results[col].isnull().sum()
        if null_count > 0:
            print(f"⚠️  特征 {col} 有 {null_count} 个空值")
        else:
            print(f"✅ 特征 {col} 无空值")
    
    print("\n✅ Week 1集成测试通过!")
    return True


def test_czsc_feature_quality():
    """测试CZSC特征质量"""
    print("\n" + "="*60)
    print("CZSC特征质量测试")
    print("="*60)
    
    # 创建更长的测试数据
    dates = pd.date_range('2023-01-01', periods=250)
    np.random.seed(123)
    
    # 生成趋势明显的数据
    base = 10
    trend = np.linspace(0, 5, 250)
    noise = np.random.randn(250) * 0.3
    price = base + trend + noise
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': price + np.random.randn(250) * 0.1,
        'close': price + np.random.randn(250) * 0.1,
        'high': price + abs(np.random.randn(250) * 0.2),
        'low': price - abs(np.random.randn(250) * 0.2),
        'volume': np.random.randint(1000, 10000, 250),
        'symbol': 'TEST'
    })
    
    generator = CzscFeatureGenerator()
    result = generator.generate_features(df)
    
    # 验证特征合理性
    print(f"\n特征统计:")
    print(f"  数据长度: {len(result)}")
    print(f"  分型数量: {(result['fx_mark'] != 0).sum()}")
    print(f"  顶分型: {(result['fx_mark'] == 1).sum()}")
    print(f"  底分型: {(result['fx_mark'] == -1).sum()}")
    print(f"  上涨笔: {(result['bi_direction'] == 1).sum()}")
    print(f"  下跌笔: {(result['bi_direction'] == -1).sum()}")
    
    # 验证笔位置在0-1之间
    bi_pos_valid = ((result['bi_position'] >= 0) & (result['bi_position'] <= 1)).all()
    assert bi_pos_valid, "笔位置应在0-1之间"
    print(f"✅ 笔位置范围正确 [0-1]")
    
    # 验证分型标记只有-1,0,1
    fx_valid = result['fx_mark'].isin([-1, 0, 1]).all()
    assert fx_valid, "分型标记应为-1,0,1"
    print(f"✅ 分型标记值正确 [-1,0,1]")
    
    print("\n✅ 特征质量测试通过!")
    return True


if __name__ == '__main__':
    try:
        test_czsc_handler_mock()
        test_czsc_feature_quality()
        print("\n" + "="*60)
        print("🎉 所有Week 1集成测试通过!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise
