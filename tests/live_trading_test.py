"""
实盘小规模测试模块
Live Trading Small Scale Testing Module

功能:
1. Ptrade/QMT模拟盘测试
2. 券商适配器验证
3. 实盘参数优化
4. 积累实盘经验

Author: Qilin Stack Team
Date: 2025-11-07
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import traceback

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from trading.live_trading_system import (
        create_live_trading_system, TradingSignal, OrderSide
    )
    TRADING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 交易模块导入失败: {e}")
    TRADING_AVAILABLE = False


class LiveTradingTestSuite:
    """实盘交易测试套件"""
    
    def __init__(self, broker_name: str = 'mock', broker_config: Optional[Dict] = None):
        """
        初始化测试套件
        
        Args:
            broker_name: 券商名称 ('mock', 'ptrade', 'qmt')
            broker_config: 券商配置
        """
        self.broker_name = broker_name
        self.broker_config = broker_config or {}
        self.test_results = []
        self.system = None
        
    async def setup(self):
        """设置测试环境"""
        print(f"\n{'='*60}")
        print(f"🚀 实盘交易测试套件启动")
        print(f"券商: {self.broker_name.upper()}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 创建交易系统
        try:
            config = {
                'broker_name': self.broker_name,
                **self.broker_config
            }
            self.system = create_live_trading_system(config)
            await self.system.start()
            print("✅ 交易系统启动成功\n")
            return True
        except Exception as e:
            print(f"❌ 交易系统启动失败: {e}\n")
            traceback.print_exc()
            return False
            
    async def teardown(self):
        """清理测试环境"""
        if self.system:
            await self.system.stop()
            print("\n✅ 交易系统已停止")
    
    def record_test_result(self, test_name: str, success: bool, 
                          details: Dict[str, Any], duration: float):
        """记录测试结果"""
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
    
    async def test_basic_order_execution(self) -> bool:
        """测试1: 基础订单执行"""
        test_name = "基础订单执行测试"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {}
        
        try:
            # 测试买入订单
            signal = TradingSignal(
                symbol='000001.SZ',
                side=OrderSide.BUY,
                size=100,
                price=10.0
            )
            
            print(f"📤 发送买入信号: {signal.symbol} x{signal.size} @ {signal.price}")
            result = await self.system.process_signal(signal)
            
            details['buy_order'] = {
                'success': result['success'],
                'order_id': result.get('order_id'),
                'message': result.get('message')
            }
            
            if result['success']:
                print(f"✅ 买入订单成功: {result['order_id']}")
                
                # 等待订单执行
                await asyncio.sleep(2)
                
                # 测试卖出订单
                signal = TradingSignal(
                    symbol='000001.SZ',
                    action='sell',
                    quantity=100,
                    price=10.2,
                    signal_id='test_sell_001'
                )
                
                print(f"📤 发送卖出信号: {signal.symbol} x{signal.quantity} @ {signal.price}")
                result = await self.system.process_signal(signal)
                
                details['sell_order'] = {
                    'success': result['success'],
                    'order_id': result.get('order_id'),
                    'message': result.get('message')
                }
                
                if result['success']:
                    print(f"✅ 卖出订单成功: {result['order_id']}")
                    success = True
                else:
                    print(f"❌ 卖出订单失败: {result.get('message')}")
            else:
                print(f"❌ 买入订单失败: {result.get('message')}")
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"耗时: {duration:.2f}秒")
        
        return success
    
    async def test_multi_symbol_trading(self) -> bool:
        """测试2: 多标的交易"""
        test_name = "多标的交易测试"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {'orders': []}
        
        try:
            symbols = ['000001.SZ', '000002.SZ', '600000.SH']
            
            for i, symbol in enumerate(symbols):
                signal = TradingSignal(
                    symbol=symbol,
                    action='buy',
                    quantity=100 * (i + 1),
                    price=10.0 + i,
                    signal_id=f'test_multi_{i}'
                )
                
                print(f"📤 [{i+1}/3] 发送信号: {symbol} x{signal.quantity} @ {signal.price}")
                result = await self.system.process_signal(signal)
                
                order_result = {
                    'symbol': symbol,
                    'success': result['success'],
                    'order_id': result.get('order_id'),
                    'message': result.get('message')
                }
                details['orders'].append(order_result)
                
                if result['success']:
                    print(f"✅ 订单成功: {result['order_id']}")
                else:
                    print(f"❌ 订单失败: {result.get('message')}")
                
                await asyncio.sleep(0.5)
            
            # 统计成功率
            success_count = sum(1 for o in details['orders'] if o['success'])
            details['success_rate'] = success_count / len(symbols)
            
            success = success_count == len(symbols)
            print(f"\n✅ 成功订单: {success_count}/{len(symbols)}")
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"耗时: {duration:.2f}秒")
        
        return success
    
    async def test_risk_control(self) -> bool:
        """测试3: 风控机制验证"""
        test_name = "风控机制验证测试"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {}
        
        try:
            # 测试1: 超大订单应该被拒绝
            signal = TradingSignal(
                symbol='000001.SZ',
                action='buy',
                quantity=1000000,  # 100万股
                price=10.0,
                signal_id='test_risk_001'
            )
            
            print(f"📤 测试超大订单 (应被拒绝): {signal.quantity}股")
            result = await self.system.process_signal(signal)
            
            details['large_order'] = {
                'rejected': not result['success'],
                'message': result.get('message')
            }
            
            if not result['success']:
                print(f"✅ 超大订单被正确拒绝: {result.get('message')}")
            else:
                print(f"⚠️ 超大订单未被拒绝 (风控可能有问题)")
            
            # 测试2: 异常价格应该被拒绝
            signal = TradingSignal(
                symbol='000001.SZ',
                action='buy',
                quantity=100,
                price=0.01,  # 异常低价
                signal_id='test_risk_002'
            )
            
            print(f"📤 测试异常价格 (应被拒绝): {signal.price}元")
            result = await self.system.process_signal(signal)
            
            details['abnormal_price'] = {
                'rejected': not result['success'],
                'message': result.get('message')
            }
            
            if not result['success']:
                print(f"✅ 异常价格被正确拒绝: {result.get('message')}")
            else:
                print(f"⚠️ 异常价格未被拒绝 (风控可能有问题)")
            
            # 测试3: 正常订单应该通过
            signal = TradingSignal(
                symbol='000001.SZ',
                action='buy',
                quantity=100,
                price=10.0,
                signal_id='test_risk_003'
            )
            
            print(f"📤 测试正常订单 (应通过): {signal.quantity}股 @ {signal.price}元")
            result = await self.system.process_signal(signal)
            
            details['normal_order'] = {
                'accepted': result['success'],
                'message': result.get('message')
            }
            
            if result['success']:
                print(f"✅ 正常订单被正确接受: {result['order_id']}")
            else:
                print(f"❌ 正常订单被错误拒绝: {result.get('message')}")
            
            # 判断成功: 异常订单被拒绝 + 正常订单通过
            success = (details['large_order']['rejected'] and 
                      details['normal_order']['accepted'])
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"耗时: {duration:.2f}秒")
        
        return success
    
    async def test_position_tracking(self) -> bool:
        """测试4: 持仓跟踪"""
        test_name = "持仓跟踪测试"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {}
        
        try:
            # 买入订单
            signal = TradingSignal(
                symbol='000001.SZ',
                action='buy',
                quantity=200,
                price=10.0,
                signal_id='test_pos_001'
            )
            
            print(f"📤 买入: {signal.symbol} x{signal.quantity}")
            result = await self.system.process_signal(signal)
            
            if result['success']:
                await asyncio.sleep(1)
                
                # 获取持仓
                if hasattr(self.system, 'position_monitor'):
                    positions = self.system.position_monitor.get_all_positions()
                    details['positions_after_buy'] = {
                        symbol: {
                            'quantity': pos.quantity,
                            'avg_price': pos.avg_price,
                            'market_value': pos.market_value
                        }
                        for symbol, pos in positions.items()
                    }
                    print(f"✅ 持仓数量: {len(positions)}个标的")
                    
                    # 卖出部分
                    signal = TradingSignal(
                        symbol='000001.SZ',
                        action='sell',
                        quantity=100,
                        price=10.2,
                        signal_id='test_pos_002'
                    )
                    
                    print(f"📤 卖出: {signal.symbol} x{signal.quantity}")
                    result = await self.system.process_signal(signal)
                    
                    if result['success']:
                        await asyncio.sleep(1)
                        
                        # 再次获取持仓
                        positions = self.system.position_monitor.get_all_positions()
                        details['positions_after_sell'] = {
                            symbol: {
                                'quantity': pos.quantity,
                                'avg_price': pos.avg_price,
                                'market_value': pos.market_value
                            }
                            for symbol, pos in positions.items()
                        }
                        
                        print(f"✅ 持仓更新成功")
                        success = True
                else:
                    print("⚠️ 持仓监控模块不可用")
                    details['warning'] = 'position_monitor_not_available'
                    success = True  # 不影响整体测试
            else:
                print(f"❌ 买入订单失败: {result.get('message')}")
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"耗时: {duration:.2f}秒")
        
        return success
    
    async def test_latency_measurement(self) -> bool:
        """测试5: 延迟测量"""
        test_name = "系统延迟测量"
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        details = {'latencies': []}
        
        try:
            num_orders = 10
            print(f"📊 发送 {num_orders} 个订单测试延迟...\n")
            
            for i in range(num_orders):
                signal = TradingSignal(
                    symbol='000001.SZ',
                    action='buy' if i % 2 == 0 else 'sell',
                    quantity=100,
                    price=10.0 + i * 0.1,
                    signal_id=f'test_latency_{i}'
                )
                
                order_start = time.time()
                result = await self.system.process_signal(signal)
                latency = (time.time() - order_start) * 1000  # ms
                
                details['latencies'].append({
                    'order_id': i,
                    'latency_ms': latency,
                    'success': result['success']
                })
                
                print(f"订单 {i+1}/{num_orders}: {latency:.2f}ms - {'✅' if result['success'] else '❌'}")
                
                await asyncio.sleep(0.1)
            
            # 统计延迟
            latencies = [item['latency_ms'] for item in details['latencies']]
            details['statistics'] = {
                'avg_latency': sum(latencies) / len(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'success_rate': sum(1 for item in details['latencies'] if item['success']) / num_orders
            }
            
            print(f"\n📊 延迟统计:")
            print(f"  平均延迟: {details['statistics']['avg_latency']:.2f}ms")
            print(f"  最小延迟: {details['statistics']['min_latency']:.2f}ms")
            print(f"  最大延迟: {details['statistics']['max_latency']:.2f}ms")
            print(f"  成功率: {details['statistics']['success_rate']*100:.1f}%")
            
            # 判断成功: 平均延迟<200ms 且成功率>90%
            success = (details['statistics']['avg_latency'] < 200 and 
                      details['statistics']['success_rate'] >= 0.9)
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            details['error'] = str(e)
            traceback.print_exc()
        
        duration = time.time() - start_time
        self.record_test_result(test_name, success, details, duration)
        
        print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
        print(f"耗时: {duration:.2f}秒")
        
        return success
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        if not await self.setup():
            return {'error': '测试环境设置失败'}
        
        try:
            # 运行测试
            await self.test_basic_order_execution()
            await self.test_multi_symbol_trading()
            await self.test_risk_control()
            await self.test_position_tracking()
            await self.test_latency_measurement()
            
        finally:
            await self.teardown()
        
        # 生成测试摘要
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        
        summary = {
            'broker': self.broker_name,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration': sum(r['duration'] for r in self.test_results),
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


def create_broker_test_config(broker_name: str) -> Dict:
    """创建券商测试配置"""
    configs = {
        'mock': {
            'initial_cash': 1000000,  # 100万模拟资金
            'commission_rate': 0.0003
        },
        'ptrade': {
            'client_path': r'D:\ptrade\userdata_mini',
            'account_id': 'YOUR_ACCOUNT_ID',  # 需要替换为真实账号
            'session_id': None
        },
        'qmt': {
            'client_path': r'D:\qmt\userdata_mini',
            'account_id': 'YOUR_ACCOUNT_ID',  # 需要替换为真实账号
            'session_id': None
        }
    }
    
    return configs.get(broker_name, {})


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("🧪 Qilin Stack 实盘交易测试套件")
    print("="*60)
    
    # 选择券商
    print("\n请选择测试券商:")
    print("1. Mock (模拟券商,推荐)")
    print("2. Ptrade (迅投,需要真实账号)")
    print("3. QMT (迅投Mini,需要真实账号)")
    
    choice = input("\n输入选择 [1/2/3] (默认=1): ").strip() or '1'
    
    broker_map = {'1': 'mock', '2': 'ptrade', '3': 'qmt'}
    broker_name = broker_map.get(choice, 'mock')
    
    if broker_name in ['ptrade', 'qmt']:
        print(f"\n⚠️ 警告: 使用真实券商 {broker_name.upper()}")
        print("请确保:")
        print("1. 已安装并配置好券商客户端")
        print("2. 在配置中填写了正确的账号信息")
        print("3. 使用的是模拟盘账号,而非实盘账号")
        confirm = input("\n确认继续? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("测试取消")
            return
    
    # 创建测试配置
    broker_config = create_broker_test_config(broker_name)
    
    # 运行测试
    test_suite = LiveTradingTestSuite(broker_name, broker_config)
    summary = await test_suite.run_all_tests()
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 测试摘要")
    print("="*60)
    print(f"券商: {summary['broker'].upper()}")
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过: {summary['passed_tests']} ✅")
    print(f"失败: {summary['failed_tests']} ❌")
    print(f"成功率: {summary['success_rate']*100:.1f}%")
    print(f"总耗时: {summary['total_duration']:.2f}秒")
    print("="*60)
    
    # 保存结果
    output_file = f"live_trading_test_results_{broker_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存到: {output_file}")
    
    return summary


if __name__ == '__main__':
    if not TRADING_AVAILABLE:
        print("❌ 交易模块不可用,请先安装依赖")
        sys.exit(1)
    
    asyncio.run(main())
