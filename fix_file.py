# 读取备份文件
with open('strategies/multi_agent_selector.py.backup', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到并删除第47-64行（索引46-63）
# 这些行是残留的VolumeAgent代码片段
new_lines = lines[:46] + lines[64:]

# 写入新文件
with open('strategies/multi_agent_selector.py', 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(new_lines)

print(f'原始行数: {len(lines)}')
print(f'删除后行数: {len(new_lines)}')
print('修复完成')
