#!/usr/bin/env python
"""
批量修复RD-Agent中的语法错误（括号未闭合问题）
"""
import re
from pathlib import Path

def fix_unclosed_parenthesis(file_path: Path):
    """修复未闭合的括号"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 模式1: 变量赋值时的多行括号问题
    # xxx: Type = (
    #     value
    # 下一行
    pattern1 = re.compile(
        r'(\s+\w+:\s+[^=]+=\s*\(\s*\n\s+[^)]+\n)(\s+)(\w+)',
        re.MULTILINE
    )
    content = pattern1.sub(r'\1    )\n\2\3', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 修复: {file_path}")
        return True
    return False

def scan_and_fix(rdagent_path: str):
    """扫描并修复所有Python文件"""
    rdagent_dir = Path(rdagent_path)
    fixed_count = 0
    
    for py_file in rdagent_dir.rglob("*.py"):
        try:
            if fix_unclosed_parenthesis(py_file):
                fixed_count += 1
        except Exception as e:
            print(f"❌ 处理失败 {py_file}: {e}")
    
    print(f"\n总计修复: {fixed_count} 个文件")

if __name__ == "__main__":
    scan_and_fix("D:/test/Qlib/RD-Agent")
