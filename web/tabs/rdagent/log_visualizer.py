"""
RD-Agent 原生日志可视化
- 自动发现 trace.json
- 支持 FileStorage pkl 日志
- 时间轴可视化
- 步骤详情查看
"""

import streamlit as st
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import os

# 尝试导入上游 RD-Agent FileStorage
try:
    from rdagent.log.storage import FileStorage
    from rdagent.log.base import Message
    HAS_RDAGENT = True
except ImportError:
    HAS_RDAGENT = False
    FileStorage = None
    Message = None


# 可能的trace.json文件位置
POSSIBLE_TRACE_FILES = [
    Path.cwd() / 'workspace' / 'trace.json',
    Path.home() / '.rdagent' / 'trace.json',
]
_env_path = os.getenv('RDAGENT_PATH')
if _env_path:
    POSSIBLE_TRACE_FILES.append(Path(_env_path) / 'workspace' / 'trace.json')

# 可能的FileStorage日志目录
POSSIBLE_LOG_DIRS = [
    Path.cwd() / 'workspace' / 'log',
    Path.cwd() / 'log',
    Path.home() / '.rdagent' / 'log',
]
if _env_path:
    POSSIBLE_LOG_DIRS.append(Path(_env_path) / 'workspace' / 'log')
    POSSIBLE_LOG_DIRS.append(Path(_env_path) / 'log')


def _load_traces_from_json(trace_path: Optional[Path]) -> List[Dict[str, Any]]:
    """从 trace.json 读取日志"""
    candidates = [trace_path] if trace_path else POSSIBLE_TRACE_FILES
    for p in candidates:
        try:
            if not p:
                continue
            if p.exists():
                text = p.read_text(encoding='utf-8')
                data = json.loads(text)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'traces' in data:
                    return data['traces']
                else:
                    return [data]
        except Exception:
            continue
    return []


def _load_traces_from_filestorage_upstream(log_dir: Path) -> List[Dict[str, Any]]:
    """使用上游 FileStorage 直接读取 pkl 日志（优先级1）"""
    if not HAS_RDAGENT or not log_dir or not log_dir.exists():
        return []
    
    traces = []
    try:
        storage = FileStorage(log_dir)
        messages = list(storage.iter_msg())
        
        # 转换 Message 对象为 trace 格式
        for msg in messages:
            traces.append({
                'id': msg.timestamp.strftime('%Y%m%d_%H%M%S_%f'),
                'type': msg.tag.split('.')[-1] if msg.tag else 'Unknown',
                'stage': msg.tag or 'Unknown',
                'status': 'completed',
                'timestamp': msg.timestamp,
                'duration': 0,
                'description': str(msg.content)[:200] if msg.content else '',
                'metadata': {
                    'tag': msg.tag,
                    'pid_trace': msg.pid_trace,
                    'level': msg.level,
                },
                'result': {'content': msg.content},
                'content': msg.content
            })
        
        st.success(f"✅ 使用上游 FileStorage 读取到 {len(traces)} 条日志")
        return traces
        
    except Exception as e:
        st.warning(f"⚠️ FileStorage 读取失败，回退到本地扫描: {e}")
        return []


def _load_traces_from_filestorage(log_dir: Path, tag_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """本地扫描 pkl 文件（优先级2，兜底方案）"""
    traces = []
    if not log_dir or not log_dir.exists():
        return traces
    
    try:
        # 搜索所有pkl文件
        pattern = f"**/{tag_filter.replace('.','/')}/**/*.pkl" if tag_filter else "**/*.pkl"
        pkl_files = list(log_dir.glob(pattern))
        
        for file in pkl_files:
            if file.name == "debug_llm.pkl":
                continue
            try:
                # 解析tag（从相对路径）
                rel_path = file.relative_to(log_dir)
                pkl_log_tag = ".".join(rel_path.as_posix().replace("/", ".").split(".")[:-3])
                pid = file.parent.name
                
                # 解析时间戳
                timestamp = datetime.strptime(file.stem, "%Y-%m-%d_%H-%M-%S-%f").replace(tzinfo=timezone.utc)
                
                # 加载pkl内容
                with file.open("rb") as f:
                    content = pickle.load(f)
                
                # 转换为trace格式
                traces.append({
                    'id': file.stem,
                    'type': pkl_log_tag.split('.')[-1] if pkl_log_tag else 'Unknown',
                    'stage': pkl_log_tag,
                    'status': 'completed',
                    'timestamp': timestamp.isoformat(),
                    'duration': 0,
                    'description': str(content)[:200] if content else '',
                    'metadata': {
                        'pid_trace': pid,
                        'file': str(file),
                        'tag': pkl_log_tag
                    },
                    'result': {'content': content},
                    'content': content
                })
            except Exception as e:
                st.warning(f"跳过文件 {file.name}: {e}")
                continue
        
        # 按时间排序
        traces.sort(key=lambda x: x['timestamp'])
        
    except Exception as e:
        st.error(f"读取FileStorage日志失败: {e}")
    
    return traces


def _normalize(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm = []
    for i, t in enumerate(traces):
        try:
            ts = t.get('timestamp') or t.get('created_at') or datetime.now().isoformat()
            try:
                # tolerant parse
                ts_dt = datetime.fromisoformat(ts.replace('Z','').split('+')[0])
            except Exception:
                ts_dt = datetime.now()
            norm.append({
                'id': t.get('id', t.get('trace_id', f'trace_{i}')),
                'stage': t.get('type', t.get('stage', 'Unknown')),
                'status': t.get('status', 'completed'),
                'timestamp': ts_dt,
                'duration': float(t.get('duration', 0) or 0),
                'description': t.get('description', t.get('task', '')),
                'metadata': t.get('metadata', t.get('details', {})),
                'result': t.get('result', {})
            })
        except Exception:
            pass
    # sort by time
    norm.sort(key=lambda x: x['timestamp'])
    return norm


def _render_token_statistics(items: List[Dict[str, Any]]):
    """渲染 Token 成本统计信息"""
    st.subheader('💰 Token 成本统计')
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    token_details_by_stage = {}
    
    # 遍历所有 items 提取 token 信息
    for item in items:
        metadata = item.get('metadata', {})
        stage = item.get('stage', 'Unknown')
        
        # 提取 token 数据（支持多种格式）
        tokens_data = None
        
        # 格式1: metadata['tokens'] 或 metadata['token_usage']
        if 'tokens' in metadata:
            tokens_data = metadata['tokens']
        elif 'token_usage' in metadata:
            tokens_data = metadata['token_usage']
        
        # 格式2: 直接在 metadata 中
        if not tokens_data and any(k in metadata for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']):
            tokens_data = metadata
        
        if tokens_data:
            prompt = tokens_data.get('prompt_tokens', 0)
            completion = tokens_data.get('completion_tokens', 0)
            total = tokens_data.get('total_tokens', prompt + completion)
            
            total_prompt_tokens += prompt
            total_completion_tokens += completion
            total_tokens += total
            
            # 按阶段统计
            if stage not in token_details_by_stage:
                token_details_by_stage[stage] = {'prompt': 0, 'completion': 0, 'total': 0, 'count': 0}
            token_details_by_stage[stage]['prompt'] += prompt
            token_details_by_stage[stage]['completion'] += completion
            token_details_by_stage[stage]['total'] += total
            token_details_by_stage[stage]['count'] += 1
    
    # 显示总体统计
    if total_tokens > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('💬 Prompt Tokens', f'{total_prompt_tokens:,}')
        with col2:
            st.metric('✅ Completion Tokens', f'{total_completion_tokens:,}')
        with col3:
            st.metric('📊 Total Tokens', f'{total_tokens:,}')
        with col4:
            # 估算成本（以 GPT-4 价格为例：$0.03/1K prompt, $0.06/1K completion）
            estimated_cost = (total_prompt_tokens / 1000 * 0.03) + (total_completion_tokens / 1000 * 0.06)
            st.metric('💵 估算成本 (USD)', f'${estimated_cost:.4f}')
        
        st.caption('💡 成本估算基于 GPT-4 定价，实际成本取决于使用的具体模型')
        
        # 按阶段分解
        if token_details_by_stage:
            with st.expander('📈 按阶段分解'):
                import pandas as pd
                df_data = []
                for stage, data in token_details_by_stage.items():
                    stage_cost = (data['prompt'] / 1000 * 0.03) + (data['completion'] / 1000 * 0.06)
                    df_data.append({
                        '阶段': stage,
                        '调用次数': data['count'],
                        'Prompt Tokens': f"{data['prompt']:,}",
                        'Completion Tokens': f"{data['completion']:,}",
                        'Total Tokens': f"{data['total']:,}",
                        '估算成本 (USD)': f"${stage_cost:.4f}"
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info('🚨 当前日志中未找到 Token 统计信息')
        st.caption('ℹ️ Token 统计仅在使用 FileStorage 记录的日志中可用')


def _render_timeline(items: List[Dict[str, Any]]):
    if not items:
        st.info('未找到trace记录')
        return
    stages = list({it['stage'] for it in items})
    fig = go.Figure()
    color_map = {
        'Research': 'blue',
        'Development': 'orange',
        'Experiment': 'green',
        'Evaluation': 'purple',
        'Unknown': 'gray'
    }
    for s in stages:
        xs = [it['timestamp'] for it in items if it['stage'] == s]
        ys = [s] * len(xs)
        texts = [it['description'] or it['status'] for it in items if it['stage'] == s]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name=s, marker=dict(size=10, color=color_map.get(s, 'gray')), text=texts))
    fig.update_layout(title='RD-Agent 执行时间轴', xaxis_title='时间', yaxis_title='阶段', height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_detail(items: List[Dict[str, Any]]):
    ids = [it['id'] for it in items]
    sel = st.selectbox('选择步骤ID', ids)
    it = next((x for x in items if x['id'] == sel), None)
    if not it:
        return
    st.subheader('步骤详情')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('阶段', it['stage'])
    with c2:
        st.metric('状态', it['status'])
    with c3:
        st.metric('耗时(s)', f"{it['duration']}")
    st.text_area('描述', it.get('description',''), height=80)
    with st.expander('元数据'):
        st.json(it.get('metadata', {}))
    with st.expander('结果'):
        st.json(it.get('result', {}))


def render():
    st.title('🧾 RD-Agent 原生日志可视化')
    st.caption('支持 trace.json 和 FileStorage pkl 日志')

    # 选择日志源类型
    log_source = st.radio("日志源类型", ['trace.json', 'FileStorage (目录)'], horizontal=True)
    
    traces = []
    
    if log_source == 'trace.json':
        # trace.json模式
        default = ''
        for p in POSSIBLE_TRACE_FILES:
            if p.exists():
                default = str(p)
                break
        user_path = st.text_input('trace.json 路径(可选)', value=default)
        trace_path = Path(user_path) if user_path else None
        traces = _load_traces_from_json(trace_path)
    else:
        # FileStorage目录模式 - 三级优先级
        default_dir = ''
        for p in POSSIBLE_LOG_DIRS:
            if p.exists():
                default_dir = str(p)
                break
        log_dir_path = st.text_input('FileStorage 日志目录', value=default_dir, help='输入包含 pkl 文件的目录路径')
        tag_filter = st.text_input('标签过滤(可选)', value='', help='例如: loop.step')
        
        if log_dir_path:
            log_dir = Path(log_dir_path)
            
            # 优先级1: 上游 FileStorage
            traces = _load_traces_from_filestorage_upstream(log_dir)
            
            # 优先级2: 本地 pkl 扫描
            if not traces:
                traces = _load_traces_from_filestorage(log_dir, tag_filter if tag_filter.strip() else None)
            
            # 优先级3: trace.json 兜底
            if not traces:
                trace_json = log_dir / 'trace.json'
                if trace_json.exists():
                    st.info("🔄 回退到 trace.json 模式")
                    traces = _load_traces_from_json(trace_json)

    # 加载并规范化数据
    items = _normalize(traces)
    
    # Token 成本统计（优先级最高）
    if items:
        _render_token_statistics(items)
        st.divider()
    
    # 过滤器
    col1, col2 = st.columns(2)
    with col1:
        stage_filter = st.multiselect('阶段过滤', ['Research','Development','Experiment','Evaluation'])
    with col2:
        status_filter = st.multiselect('状态过滤', ['success','failed','running','completed'])

    # 应用过滤
    if stage_filter:
        items = [x for x in items if x['stage'] in stage_filter]
    if status_filter:
        items = [x for x in items if (x['status'] or '').lower() in {s.lower() for s in status_filter}]
    
    st.info(f"共找到 {len(items)} 条日志记录")

    _render_timeline(items)
    st.divider()
    _render_detail(items)


if __name__ == '__main__':
    render()
