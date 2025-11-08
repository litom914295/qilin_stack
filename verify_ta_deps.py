import sys
import importlib
from importlib import metadata

print('Python:', sys.version)

# Optional: allow passing external path via PYTHONPATH if needed
print('PYTHONPATH:', sys.path[:2])

# Print versions
pkgs = [
    'langchain','langchain-core','langchain-openai','langchain-community',
    'langchain-text-splitters','langsmith','langgraph','langgraph-prebuilt',
]
for p in pkgs:
    try:
        print(p, metadata.version(p))
    except Exception:
        print(p, 'not installed')

# Import checks
try:
    import tradingagents
    print('tradingagents OK')
except Exception as e:
    print('tradingagents ERROR:', type(e).__name__, str(e))

try:
    from langchain.agents import create_react_agent, AgentExecutor
    print('langchain.agents OK')
except Exception as e:
    print('langchain.agents ERROR:', type(e).__name__, str(e))

try:
    from langgraph.prebuilt import ToolNode
    print('langgraph.prebuilt OK')
except Exception as e:
    print('langgraph.prebuilt ERROR:', type(e).__name__, str(e))

# Try to import qilin_agents
try:
    importlib.import_module('tradingagents.agents.qilin_agents')
    print('qilin_agents OK')
except Exception as e:
    print('qilin_agents ERROR:', type(e).__name__, str(e))
