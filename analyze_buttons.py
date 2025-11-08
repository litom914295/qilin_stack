#!/usr/bin/env python3
"""
分析项目中的 st.button 调用，查找潜在的重复按钮问题
"""
import re
from pathlib import Path
from collections import defaultdict

def extract_button_info(file_path):
    """从文件中提取所有 st.button 调用信息"""
    buttons = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # 匹配 st.button 调用
                match = re.search(r'st\.button\s*\(\s*["\']([^"\']+)["\']([^)]*)\)', line)
                if match:
                    button_text = match.group(1)
                    params = match.group(2)
                    
                    # 检查是否有 key 参数
                    has_key = 'key=' in params
                    
                    buttons.append({
                        'file': str(file_path),
                        'line': line_num,
                        'text': button_text,
                        'has_key': has_key,
                        'full_line': line.strip()
                    })
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return buttons

def analyze_buttons():
    """分析项目中的所有按钮"""
    project_root = Path(r'G:\test\qilin_stack\web')
    all_buttons = []
    
    # 遍历所有 Python 文件
    for py_file in project_root.rglob('*.py'):
        buttons = extract_button_info(py_file)
        all_buttons.extend(buttons)
    
    print(f"✅ 总共找到 {len(all_buttons)} 个按钮\n")
    
    # 按文本分组
    buttons_by_text = defaultdict(list)
    for btn in all_buttons:
        buttons_by_text[btn['text']].append(btn)
    
    # 查找重复的按钮文本
    print("=" * 80)
    print("🔍 重复的按钮文本（可能存在ID冲突风险）:")
    print("=" * 80)
    
    duplicates_found = False
    for text, buttons in sorted(buttons_by_text.items()):
        if len(buttons) > 1:
            duplicates_found = True
            print(f"\n📌 按钮文本: '{text}' - 出现 {len(buttons)} 次")
            
            # 检查是否所有实例都有唯一的 key
            without_key = [b for b in buttons if not b['has_key']]
            
            if without_key:
                print(f"   ⚠️  警告: {len(without_key)} 个实例没有 key 参数（可能导致冲突）")
                for btn in without_key:
                    rel_path = btn['file'].replace('G:\\test\\qilin_stack\\', '')
                    print(f"      - {rel_path}:{btn['line']}")
            else:
                print(f"   ✅ 所有实例都有唯一的 key 参数")
    
    if not duplicates_found:
        print("\n✅ 没有发现重复的按钮文本!")
    
    # 统计没有 key 的按钮
    print("\n" + "=" * 80)
    print("📊 没有 key 参数的按钮统计:")
    print("=" * 80)
    
    no_key_buttons = [b for b in all_buttons if not b['has_key']]
    print(f"\n总共有 {len(no_key_buttons)} 个按钮没有 key 参数 ({len(no_key_buttons)/len(all_buttons)*100:.1f}%)\n")
    
    # 按文件分组显示
    by_file = defaultdict(list)
    for btn in no_key_buttons:
        by_file[btn['file']].append(btn)
    
    for file_path, buttons in sorted(by_file.items())[:10]:  # 只显示前10个文件
        rel_path = file_path.replace('G:\\test\\qilin_stack\\', '')
        print(f"\n📄 {rel_path} ({len(buttons)} 个按钮)")
        for btn in buttons[:5]:  # 每个文件最多显示5个
            print(f"   行 {btn['line']:4d}: {btn['text']}")
        if len(buttons) > 5:
            print(f"   ... 还有 {len(buttons) - 5} 个")
    
    # 高风险文件（同一文件中有多个相同文本的按钮且没有key）
    print("\n" + "=" * 80)
    print("⚠️  高风险文件（同一文件内有重复按钮文本且无key）:")
    print("=" * 80)
    
    high_risk_found = False
    for file_path, buttons in by_file.items():
        # 按文本分组
        text_groups = defaultdict(list)
        for btn in buttons:
            text_groups[btn['text']].append(btn)
        
        # 找出有重复的
        duplicates_in_file = {text: btns for text, btns in text_groups.items() if len(btns) > 1}
        
        if duplicates_in_file:
            high_risk_found = True
            rel_path = file_path.replace('G:\\test\\qilin_stack\\', '')
            print(f"\n📄 {rel_path}")
            for text, btns in duplicates_in_file.items():
                print(f"   ❌ '{text}' 出现 {len(btns)} 次:")
                for btn in btns:
                    print(f"      行 {btn['line']}: {btn['full_line'][:80]}")
    
    if not high_risk_found:
        print("\n✅ 没有发现高风险文件!")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成!")
    print("=" * 80)

if __name__ == '__main__':
    analyze_buttons()
