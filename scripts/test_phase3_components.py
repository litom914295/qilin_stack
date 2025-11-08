"""
Phase 3组件测试脚本
测试UI优化、加载动画、缓存、快捷键、智能提示等功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

# 导入Phase 3组件
from web.components.color_scheme import (
    Colors, Emojis, get_strength_color, get_strength_emoji,
    get_profit_color, get_status_badge_html, get_progress_bar_html
)
from web.components.ui_styles import inject_global_styles, create_section_header
from web.components.loading_cache import (
    LoadingSpinner, CacheManager, PerformanceMonitor, LazyLoader
)
from web.components.keyboard_shortcuts import KeyboardShortcuts
from web.components.smart_tips_enhanced import EnhancedSmartTipSystem


def test_color_scheme():
    """测试颜色编码系统"""
    print("=" * 60)
    print("测试 1: 颜色编码系统")
    print("=" * 60)
    
    try:
        # 测试颜色常量
        assert hasattr(Colors, 'SUCCESS')
        assert hasattr(Colors, 'WARNING')
        assert hasattr(Colors, 'DANGER')
        print("✅ 颜色常量定义正确")
        
        # 测试Emoji常量
        assert Emojis.GREEN_CIRCLE == "🟢"
        assert Emojis.FIRE == "🔥"
        print("✅ Emoji常量定义正确")
        
        # 测试强度颜色映射
        color_9 = get_strength_color(9.5)
        assert color_9 == Colors.STRONG_GREEN
        print(f"✅ 强度9.5 → {color_9}")
        
        # 测试强度Emoji映射
        emoji_9 = get_strength_emoji(9.5)
        assert Emojis.GREEN_CIRCLE in emoji_9
        print(f"✅ 强度9.5 → {emoji_9}")
        
        # 测试盈亏颜色
        profit_color = get_profit_color(15)
        assert profit_color == Colors.STRONG_GREEN
        print(f"✅ 盈利15% → {profit_color}")
        
        # 测试HTML生成
        badge_html = get_status_badge_html("测试", "success")
        assert "测试" in badge_html and "<span" in badge_html
        print("✅ 状态徽章HTML生成正确")
        
        progress_html = get_progress_bar_html(75, 100)
        assert "75" in progress_html or "width" in progress_html
        print("✅ 进度条HTML生成正确")
        
        print("✅ 通过 - 颜色编码系统\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 颜色编码系统: {e}\n")
        return False


def test_loading_cache():
    """测试加载动画和缓存系统"""
    print("=" * 60)
    print("测试 2: 加载动画和缓存系统")
    print("=" * 60)
    
    try:
        # 测试加载动画上下文管理器
        spinner = LoadingSpinner("测试加载", "⏳")
        assert spinner.message == "测试加载"
        print("✅ LoadingSpinner初始化正确")
        
        # 测试缓存管理器
        cache_manager = CacheManager()
        assert hasattr(cache_manager, 'cache_data')
        assert hasattr(cache_manager, 'cache_resource')
        print("✅ CacheManager方法完整")
        
        # 测试性能监控
        with PerformanceMonitor("测试操作", show_result=False) as pm:
            # 模拟操作
            sum([i**2 for i in range(1000)])
        
        elapsed = pm.get_elapsed_time()
        assert elapsed >= 0
        print(f"✅ PerformanceMonitor工作正常 (耗时: {elapsed:.4f}s)")
        
        # 测试懒加载
        def mock_load_func():
            return [1, 2, 3, 4, 5]
        
        lazy = LazyLoader(mock_load_func)
        assert not lazy.is_loaded()
        data = lazy.load()
        assert lazy.is_loaded()
        assert data == [1, 2, 3, 4, 5]
        print("✅ LazyLoader懒加载工作正常")
        
        print("✅ 通过 - 加载动画和缓存系统\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 加载动画和缓存系统: {e}\n")
        return False


def test_keyboard_shortcuts():
    """测试键盘快捷键系统"""
    print("=" * 60)
    print("测试 3: 键盘快捷键系统")
    print("=" * 60)
    
    try:
        # 测试快捷键管理器
        shortcuts = KeyboardShortcuts()
        assert hasattr(shortcuts, 'register')
        assert hasattr(shortcuts, 'DEFAULT_SHORTCUTS')
        print("✅ KeyboardShortcuts初始化正确")
        
        # 测试注册快捷键
        def test_callback():
            return "test"
        
        shortcuts.register('t', test_callback, "测试快捷键")
        assert 't' in shortcuts.callbacks
        print("✅ 快捷键注册成功")
        
        # 测试启用/禁用
        shortcuts.disable()
        assert not shortcuts.enabled
        shortcuts.enable()
        assert shortcuts.enabled
        print("✅ 快捷键启用/禁用功能正常")
        
        # 测试默认快捷键
        assert 'r' in shortcuts.DEFAULT_SHORTCUTS
        assert 'e' in shortcuts.DEFAULT_SHORTCUTS
        assert 's' in shortcuts.DEFAULT_SHORTCUTS
        print("✅ 默认快捷键定义完整")
        
        print("✅ 通过 - 键盘快捷键系统\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 键盘快捷键系统: {e}\n")
        return False


def test_enhanced_smart_tips():
    """测试增强版智能提示系统"""
    print("=" * 60)
    print("测试 4: 增强版智能提示系统")
    print("=" * 60)
    
    try:
        # 测试智能提示系统
        tip_system = EnhancedSmartTipSystem()
        assert hasattr(tip_system, 'risk_rules')
        assert len(tip_system.risk_rules) > 0
        print(f"✅ EnhancedSmartTipSystem初始化正确 (风险规则数: {len(tip_system.risk_rules)})")
        
        # 测试市场情绪分析
        sentiment_high = tip_system.analyze_market_sentiment(120)
        assert sentiment_high['sentiment'] == '活跃'
        assert sentiment_high['score'] == 80
        print(f"✅ 市场情绪分析 (涨停120只): {sentiment_high['sentiment']} ({sentiment_high['score']}分)")
        
        sentiment_low = tip_system.analyze_market_sentiment(25)
        assert sentiment_low['sentiment'] == '冰点'
        print(f"✅ 市场情绪分析 (涨停25只): {sentiment_low['sentiment']}")
        
        # 测试风险预警
        test_data = {
            'sector_concentration': 70,  # 触发集中度风险
            'failed_limitup_rate': 35,   # 触发炸板风险
            'index_change': -2.5         # 触发指数风险
        }
        warnings = tip_system.check_risk_warnings(test_data)
        assert len(warnings) >= 3
        print(f"✅ 风险预警检测正确 (触发{len(warnings)}条预警)")
        
        # 测试板块分析
        test_df = pd.DataFrame({
            'symbol': ['000001', '000002', '000003', '000004', '000005'],
            'sector': ['AI', 'AI', 'AI', '芯片', '新能源']
        })
        sector_analysis = tip_system.generate_sector_analysis(test_df)
        assert 'top_sectors' in sector_analysis
        assert sector_analysis['max_percentage'] == 60  # AI占60%
        print(f"✅ 板块分析正确 (最大集中度: {sector_analysis['max_percentage']:.1f}%)")
        
        # 测试时间建议
        timing = tip_system.generate_timing_advice()
        assert 'phase' in timing
        assert 'advice' in timing
        print(f"✅ 时间建议生成正确 (当前阶段: {timing['phase']})")
        
        # 测试绩效提示
        perf_data = {
            'win_rate': 65,
            'avg_profit': 3.5,
            'max_drawdown': -8
        }
        perf_tips = tip_system.generate_performance_tips(perf_data)
        assert len(perf_tips) > 0
        print(f"✅ 绩效提示生成正确 (提示数: {len(perf_tips)})")
        
        # 测试基础提示生成（继承自父类）
        tip_data = {
            'limitup_count': 80,
            'candidate_count': 8,
            'avg_quality_score': 75
        }
        tips = tip_system.generate_tips("T日选股", tip_data)
        assert len(tips) > 0
        print(f"✅ 基础提示生成正确 (提示数: {len(tips)})")
        
        print("✅ 通过 - 增强版智能提示系统\n")
        return True
        
    except Exception as e:
        print(f"❌ 失败 - 增强版智能提示系统: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Phase 3 组件测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 执行所有测试
    results.append(("颜色编码系统", test_color_scheme()))
    results.append(("加载动画和缓存", test_loading_cache()))
    results.append(("键盘快捷键", test_keyboard_shortcuts()))
    results.append(("增强版智能提示", test_enhanced_smart_tips()))
    
    # 统计结果
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有Phase 3组件测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
