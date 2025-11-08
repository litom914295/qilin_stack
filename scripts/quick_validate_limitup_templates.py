#!/usr/bin/env python
"""
快速验证新增一进二模板（GATs/SFM/TCN）
仅做冒烟测试，无需完整训练，验证配置加载和集成
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_template(template_path: Path) -> Dict:
    """
    验证单个模板配置
    
    Returns:
        Dict with keys: valid, errors, warnings, summary
    """
    result = {
        'template_name': template_path.stem,
        'valid': False,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    try:
        # 1. 读取YAML
        with open(template_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ 成功读取模板: {template_path.name}")
        
        # 2. 验证必需字段
        required_keys = ['task', 'strategy']
        for key in required_keys:
            if key not in config:
                result['errors'].append(f"缺少必需字段: {key}")
        
        if result['errors']:
            return result
        
        # 3. 提取关键配置
        task = config['task']
        model_config = task.get('model', {})
        dataset_config = task.get('dataset', {})
        strategy_config = config.get('strategy', {})
        
        result['summary'] = {
            'model_class': model_config.get('class', 'N/A'),
            'model_module': model_config.get('module_path', 'N/A'),
            'handler_class': dataset_config.get('kwargs', {}).get('handler', {}).get('class', 'N/A'),
            'instruments': dataset_config.get('kwargs', {}).get('handler', {}).get('kwargs', {}).get('instruments', 'N/A'),
            'strategy_class': strategy_config.get('class', 'N/A'),
            'topk': strategy_config.get('kwargs', {}).get('topk', 'N/A')
        }
        
        # 4. 特定模型检查
        model_class = result['summary']['model_class']
        
        if model_class == 'GATsModel':
            # GATs特定检查
            model_kwargs = model_config.get('kwargs', {})
            if 'n_heads' not in model_kwargs:
                result['warnings'].append("GATs模型缺少n_heads参数，可能影响性能")
            if 'd_feat' not in model_kwargs:
                result['errors'].append("GATs模型缺少d_feat参数（必需）")
        
        elif model_class == 'SFM':
            # SFM特定检查
            model_kwargs = model_config.get('kwargs', {})
            if 'embed_dim' not in model_kwargs:
                result['warnings'].append("SFM模型缺少embed_dim参数，将使用默认值")
            handler = result['summary']['handler_class']
            if handler != 'Alpha360':
                result['warnings'].append(f"SFM推荐使用Alpha360特征，当前使用: {handler}")
        
        elif model_class == 'TCNModel':
            # TCN特定检查
            model_kwargs = model_config.get('kwargs', {})
            if 'channels' not in model_kwargs:
                result['errors'].append("TCN模型缺少channels参数（必需）")
            if 'kernel_size' not in model_kwargs:
                result['errors'].append("TCN模型缺少kernel_size参数（必需）")
        
        # 5. 标签检查
        handler_kwargs = dataset_config.get('kwargs', {}).get('handler', {}).get('kwargs', {})
        label = handler_kwargs.get('label', '')
        
        if not label:
            result['errors'].append("缺少标签定义（label字段）")
        else:
            if '0.095' in label or '0.09' in label:
                result['summary']['label_type'] = '涨停板标签（9-10%）'
            elif 'Ref($close, -1)' in label:
                result['summary']['label_type'] = '明日收益标签'
            else:
                result['summary']['label_type'] = '自定义标签'
        
        # 6. 最终判断
        if not result['errors']:
            result['valid'] = True
            logger.info(f"✓ 模板验证通过: {template_path.name}")
        else:
            logger.error(f"✗ 模板验证失败: {template_path.name} - {result['errors']}")
        
        if result['warnings']:
            for warning in result['warnings']:
                logger.warning(f"⚠ {template_path.name}: {warning}")
        
        return result
        
    except yaml.YAMLError as e:
        result['errors'].append(f"YAML解析错误: {e}")
        logger.error(f"✗ YAML解析失败: {template_path.name}")
        return result
    
    except Exception as e:
        result['errors'].append(f"未知错误: {e}")
        logger.error(f"✗ 验证失败: {template_path.name} - {e}")
        return result


def validate_ui_integration() -> Dict:
    """
    验证UI集成（检查template_mapping）
    
    Returns:
        Dict with integration status
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'found_templates': []
    }
    
    try:
        # 读取UI文件
        ui_file = project_root / 'web' / 'tabs' / 'qlib_qrun_workflow_tab.py'
        
        if not ui_file.exists():
            result['errors'].append(f"UI文件不存在: {ui_file}")
            return result
        
        with open(ui_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查三个模板是否在映射中
        expected_mappings = {
            '行业图注意力GATs': 'limitup_gats',
            'SFM交互增强分类': 'limitup_sfm',
            '多分辨率TCN时序': 'limitup_tcn'
        }
        
        for display_name, template_id in expected_mappings.items():
            if template_id in content:
                result['found_templates'].append(template_id)
                logger.info(f"✓ UI中找到模板映射: {template_id}")
            else:
                result['errors'].append(f"UI中未找到模板映射: {template_id}")
        
        # 验证通过条件
        if len(result['found_templates']) == 3:
            result['valid'] = True
            logger.info("✓ UI集成验证通过")
        else:
            logger.error(f"✗ UI集成验证失败，仅找到 {len(result['found_templates'])}/3 个模板")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"UI集成验证失败: {e}")
        logger.error(f"✗ UI集成验证失败: {e}")
        return result


def validate_documentation() -> Dict:
    """
    验证文档是否包含新模板说明
    """
    result = {
        'valid': False,
        'errors': [],
        'found_sections': []
    }
    
    try:
        doc_file = project_root / 'docs' / 'P2-E4_LimitUp_Templates.md'
        
        if not doc_file.exists():
            result['errors'].append(f"文档不存在: {doc_file}")
            return result
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含三个模板的章节
        expected_sections = [
            'limitup_gats.yaml',
            'limitup_sfm.yaml',
            'limitup_tcn.yaml'
        ]
        
        for section in expected_sections:
            if section in content:
                result['found_sections'].append(section)
                logger.info(f"✓ 文档中找到章节: {section}")
            else:
                result['errors'].append(f"文档中未找到章节: {section}")
        
        # 验证通过条件
        if len(result['found_sections']) == 3:
            result['valid'] = True
            logger.info("✓ 文档验证通过")
        else:
            logger.error(f"✗ 文档验证失败，仅找到 {len(result['found_sections'])}/3 个章节")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"文档验证失败: {e}")
        logger.error(f"✗ 文档验证失败: {e}")
        return result


def print_summary_report(results: Dict):
    """打印汇总报告"""
    print("\n" + "="*80)
    print("                    快速验证报告")
    print("="*80)
    
    # 模板验证结果
    print("\n【模板配置验证】")
    for template_name, result in results['templates'].items():
        status = "✓ 通过" if result['valid'] else "✗ 失败"
        print(f"\n  {template_name}: {status}")
        
        if result['summary']:
            print(f"    - 模型: {result['summary'].get('model_class', 'N/A')}")
            print(f"    - 特征: {result['summary'].get('handler_class', 'N/A')}")
            print(f"    - 股票池: {result['summary'].get('instruments', 'N/A')}")
            print(f"    - 标签: {result['summary'].get('label_type', 'N/A')}")
        
        if result['errors']:
            print(f"    ✗ 错误: {', '.join(result['errors'])}")
        if result['warnings']:
            print(f"    ⚠ 警告: {', '.join(result['warnings'])}")
    
    # UI集成验证
    print("\n【UI集成验证】")
    ui_result = results['ui_integration']
    status = "✓ 通过" if ui_result['valid'] else "✗ 失败"
    print(f"  状态: {status}")
    print(f"  已集成模板: {', '.join(ui_result['found_templates'])}")
    if ui_result['errors']:
        print(f"  ✗ 错误: {', '.join(ui_result['errors'])}")
    
    # 文档验证
    print("\n【文档验证】")
    doc_result = results['documentation']
    status = "✓ 通过" if doc_result['valid'] else "✗ 失败"
    print(f"  状态: {status}")
    print(f"  已记录章节: {', '.join(doc_result['found_sections'])}")
    if doc_result['errors']:
        print(f"  ✗ 错误: {', '.join(doc_result['errors'])}")
    
    # 总体结果
    print("\n【总体结果】")
    all_valid = (
        all(r['valid'] for r in results['templates'].values()) and
        ui_result['valid'] and
        doc_result['valid']
    )
    
    if all_valid:
        print("  🎉 所有验证通过！新增3个一进二高级模板已就绪。")
        print("\n  下一步建议：")
        print("    1. 在Web UI中测试模板加载（Qlib工作流 → 从模板创建）")
        print("    2. 选择其中一个模板进行短周期训练测试（2022-2023数据）")
        print("    3. 在实验对比面板中查看结果")
    else:
        print("  ⚠ 部分验证失败，请检查上述错误并修复")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    logger.info("开始快速验证...")
    
    # 定义三个新模板路径
    template_dir = project_root / 'configs' / 'qlib_workflows' / 'templates'
    new_templates = [
        template_dir / 'limitup_gats.yaml',
        template_dir / 'limitup_sfm.yaml',
        template_dir / 'limitup_tcn.yaml'
    ]
    
    # 验证结果存储
    results = {
        'templates': {},
        'ui_integration': {},
        'documentation': {}
    }
    
    # 1. 验证每个模板
    logger.info("\n=== 验证模板配置 ===")
    for template_path in new_templates:
        if not template_path.exists():
            logger.error(f"✗ 模板文件不存在: {template_path}")
            results['templates'][template_path.stem] = {
                'valid': False,
                'errors': [f"文件不存在: {template_path}"],
                'warnings': [],
                'summary': {}
            }
        else:
            result = validate_template(template_path)
            results['templates'][template_path.stem] = result
    
    # 2. 验证UI集成
    logger.info("\n=== 验证UI集成 ===")
    results['ui_integration'] = validate_ui_integration()
    
    # 3. 验证文档
    logger.info("\n=== 验证文档 ===")
    results['documentation'] = validate_documentation()
    
    # 4. 打印汇总报告
    print_summary_report(results)
    
    # 5. 返回状态码
    all_valid = (
        all(r['valid'] for r in results['templates'].values()) and
        results['ui_integration']['valid'] and
        results['documentation']['valid']
    )
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
