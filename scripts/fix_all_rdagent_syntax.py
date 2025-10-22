#!/usr/bin/env python
"""
批量修复RD-Agent中所有的括号未闭合语法错误
"""
import re
from pathlib import Path
from typing import List, Tuple

def fix_multiline_assignment(content: str) -> Tuple[str, int]:
    """
    修复模式：
    variable: Type = (
        value
    next_line  # 缺少闭合括号
    """
    changes = 0
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检测模式：变量赋值 + 开括号
        if '=' in line and '(' in line and ')' not in line:
            # 检查是不是多行赋值的开始
            indent = len(line) - len(line.lstrip())
            potential_fix = False
            
            # 查看下一行
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_indent = len(next_line) - len(next_line.lstrip())
                
                # 如果下一行缩进更多，可能是多行赋值的内容
                if next_indent > indent:
                    # 查看再下一行
                    if i + 2 < len(lines):
                        nextnext_line = lines[i + 2]
                        nextnext_indent = len(nextnext_line) - len(nextnext_line.lstrip())
                        
                        # 如果第三行的缩进回到原始级别且不是右括号，需要修复
                        if nextnext_indent <= indent and nextnext_line.strip() and not nextnext_line.strip().startswith(')'):
                            # 需要在第二行末尾添加闭合括号
                            fixed_lines.append(line)
                            fixed_lines.append(next_line)
                            fixed_lines.append(' ' * next_indent + ')')
                            i += 2
                            changes += 1
                            potential_fix = True
            
            if not potential_fix:
                fixed_lines.append(line)
                i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    return '\n'.join(fixed_lines), changes

def process_file(file_path: Path) -> bool:
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content, changes = fix_multiline_assignment(content)
        
        if changes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✅ {file_path.relative_to('D:/test/Qlib/RD-Agent')}: 修复 {changes} 处")
            return True
        return False
    except Exception as e:
        print(f"❌ {file_path}: {e}")
        return False

def main():
    rd_agent_path = Path("D:/test/Qlib/RD-Agent")
    
    # 查找所有Python文件
    py_files = list(rd_agent_path.rglob("*.py"))
    print(f"找到 {len(py_files)} 个Python文件")
    
    fixed_count = 0
    for py_file in py_files:
        if process_file(py_file):
            fixed_count += 1
    
    print(f"\n✅ 总计修复 {fixed_count} 个文件")

if __name__ == "__main__":
    main()
