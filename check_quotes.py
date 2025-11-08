content = open('strategies/multi_agent_selector.py', 'r', encoding='utf-8').read()
triple_count = content.count('"""')
print(f"三引号数量: {triple_count}")
print(f"是否配对: {'是' if triple_count % 2 == 0 else '否'}")
if triple_count % 2 != 0:
    print("发现未配对的三引号！")
