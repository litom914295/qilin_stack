"""
配置管理器使用示例
演示如何使用ConfigManager加载和管理配置
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, get_config


def example_1_basic_usage():
    """示例1：基本使用"""
    print("\n" + "="*80)
    print("示例1：基本使用")
    print("="*80)
    
    # 创建配置管理器（使用默认配置）
    config = ConfigManager()
    
    # 访问配置项
    print(f"\n项目名称: {config.get('system.project_name')}")
    print(f"总资金: {config.get('buy.total_capital')}")
    print(f"最小封单强度: {config.get('screening.min_seal_strength')}")
    print(f"Kelly分数: {config.get('kelly.kelly_fraction')}")
    
    # 验证配置
    is_valid = config.validate()
    print(f"\n配置验证结果: {'通过' if is_valid else '失败'}")


def example_2_load_from_file():
    """示例2：从文件加载配置"""
    print("\n" + "="*80)
    print("示例2：从文件加载配置")
    print("="*80)
    
    # 从YAML文件加载配置
    config_path = "config/default_config.yaml"
    config = ConfigManager(config_file=config_path)
    
    print(f"\n配置文件: {config_path}")
    print(f"工作流自动模式: {config.get('workflow.auto_mode')}")
    print(f"启用Kelly: {config.get('kelly.enable_kelly')}")
    print(f"启用市场熔断: {config.get('market_breaker.enable_breaker')}")


def example_3_modify_config():
    """示例3：修改配置"""
    print("\n" + "="*80)
    print("示例3：修改配置")
    print("="*80)
    
    config = ConfigManager()
    
    # 修改配置项
    print(f"\n原始总资金: {config.get('buy.total_capital')}")
    config.set('buy.total_capital', 2000000)
    print(f"修改后总资金: {config.get('buy.total_capital')}")
    
    print(f"\n原始止损: {config.get('sell.stop_loss')}")
    config.set('sell.stop_loss', -0.05)
    print(f"修改后止损: {config.get('sell.stop_loss')}")


def example_4_save_config():
    """示例4：保存配置"""
    print("\n" + "="*80)
    print("示例4：保存配置")
    print("="*80)
    
    config = ConfigManager()
    
    # 修改一些配置
    config.set('buy.total_capital', 1500000)
    config.set('screening.max_candidates', 50)
    config.set('kelly.kelly_fraction', 0.3)
    
    # 保存到YAML文件
    output_yaml = "config/custom_config.yaml"
    config.save_to_file(output_yaml)
    print(f"\n✓ 配置已保存到: {output_yaml}")


def example_5_global_config():
    """示例5：使用全局配置（单例模式）"""
    print("\n" + "="*80)
    print("示例5：使用全局配置")
    print("="*80)
    
    # 获取全局配置实例
    config1 = get_config()
    config1.set('buy.total_capital', 3000000)
    
    # 再次获取，应该是同一实例
    config2 = get_config()
    
    print(f"\n配置1总资金: {config1.get('buy.total_capital')}")
    print(f"配置2总资金: {config2.get('buy.total_capital')}")
    print(f"是否为同一实例: {config1 is config2}")


def example_6_access_config_sections():
    """示例6：访问配置节"""
    print("\n" + "="*80)
    print("示例6：访问配置节")
    print("="*80)
    
    config = ConfigManager()
    
    # 获取整个配置节
    screening_config = config.get_section('screening')
    print("\n筛选配置:")
    for key, value in screening_config.items():
        print(f"  {key}: {value}")
    
    buy_config = config.get_section('buy')
    print("\n买入配置:")
    for key, value in buy_config.items():
        print(f"  {key}: {value}")
    
    kelly_config = config.get_section('kelly')
    print("\nKelly配置:")
    for key, value in kelly_config.items():
        print(f"  {key}: {value}")


def example_7_integration_with_workflow():
    """示例7：与工作流集成"""
    print("\n" + "="*80)
    print("示例7：与工作流集成")
    print("="*80)
    
    config = ConfigManager()
    
    # 读取工作流配置
    workflow_config = {
        'enable_t_day_screening': config.get('workflow.enable_t_day_screening'),
        'enable_t1_auction_monitor': config.get('workflow.enable_t1_auction_monitor'),
        'enable_t1_buy': config.get('workflow.enable_t1_buy'),
        'enable_t2_sell': config.get('workflow.enable_t2_sell'),
        'enable_journal': config.get('journal.enable_journal'),
        'enable_market_breaker': config.get('market_breaker.enable_breaker'),
        'enable_kelly_position': config.get('kelly.enable_kelly'),
        
        'screening': config.get_section('screening'),
        'auction': config.get_section('auction'),
        'buy': config.get_section('buy'),
        'sell': config.get_section('sell'),
        'risk': {
            'enable_breaker': config.get('market_breaker.enable_breaker'),
            'enable_kelly': config.get('kelly.enable_kelly')
        }
    }
    
    print("\n工作流配置已准备就绪:")
    print(f"  T日筛选: {workflow_config['enable_t_day_screening']}")
    print(f"  T+1竞价监控: {workflow_config['enable_t1_auction_monitor']}")
    print(f"  T+1买入: {workflow_config['enable_t1_buy']}")
    print(f"  T+2卖出: {workflow_config['enable_t2_sell']}")
    print(f"  Kelly仓位管理: {workflow_config['enable_kelly_position']}")
    print(f"  市场熔断: {workflow_config['enable_market_breaker']}")
    
    print(f"\n筛选参数:")
    print(f"  最小封单强度: {workflow_config['screening']['min_seal_strength']}")
    print(f"  最小预测得分: {workflow_config['screening']['min_prediction_score']}")
    print(f"  最大候选数: {workflow_config['screening']['max_candidates']}")
    
    print(f"\n买入参数:")
    print(f"  总资金: {workflow_config['buy']['total_capital']}")
    print(f"  单股最大仓位: {workflow_config['buy']['max_position_per_stock']}")
    print(f"  总仓位上限: {workflow_config['buy']['max_total_position']}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("配置管理器使用示例")
    print("="*80)
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_load_from_file()
    example_3_modify_config()
    example_4_save_config()
    example_5_global_config()
    example_6_access_config_sections()
    example_7_integration_with_workflow()
    
    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)
