"""
RD-Agent 会话管理模块
提供会话列表、控制和日志查看功能
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import threading
import time

logger = logging.getLogger(__name__)


class SessionStorage:
    """会话持久化存储(线程安全)"""
    
    def __init__(self, workspace_dir: Path = None):
        if workspace_dir is None:
            workspace_dir = Path.cwd() / "workspace" / "sessions"
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_file = self.workspace_dir / "sessions.json"
        # 线程锁: 保护文件读写操作
        self._lock = threading.Lock()
        # 日志文件锁: 按session_id分别加锁
        self._log_locks = {}  # {session_id: Lock}
    
    def load_sessions(self) -> List[Dict]:
        """加载所有会话(线程安全)"""
        with self._lock:
            if not self.sessions_file.exists():
                return []
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load sessions: {e}")
                return []
    
    def save_sessions(self, sessions: List[Dict]):
        """保存所有会话(线程安全)"""
        with self._lock:
            try:
                with open(self.sessions_file, 'w', encoding='utf-8') as f:
                    json.dump(sessions, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save sessions: {e}")
    
    def add_session(self, session: Dict):
        """添加新会话"""
        sessions = self.load_sessions()
        sessions.append(session)
        self.save_sessions(sessions)
    
    def update_session(self, session_id: str, updates: Dict):
        """更新会话"""
        sessions = self.load_sessions()
        for sess in sessions:
            if sess.get('session_id') == session_id:
                sess.update(updates)
                break
        self.save_sessions(sessions)
    
    def delete_session(self, session_id: str):
        """删除会话"""
        sessions = self.load_sessions()
        sessions = [s for s in sessions if s.get('session_id') != session_id]
        self.save_sessions(sessions)
        
        # 删除会话日志文件
        log_file = self.workspace_dir / f"{session_id}.log"
        if log_file.exists():
            log_file.unlink()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取单个会话"""
        sessions = self.load_sessions()
        for sess in sessions:
            if sess.get('session_id') == session_id:
                return sess
        return None
    
    def get_session_logs(self, session_id: str, tail: int = 100) -> List[str]:
        """获取会话日志(线程安全)"""
        log_file = self.workspace_dir / f"{session_id}.log"
        if not log_file.exists():
            return []
        
        # 获取或创建session的日志锁
        if session_id not in self._log_locks:
            self._log_locks[session_id] = threading.Lock()
        
        with self._log_locks[session_id]:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return lines[-tail:] if tail > 0 else lines
            except Exception as e:
                logger.error(f"Failed to load session logs: {e}")
                return []
    
    def get_log_path(self, session_id: str) -> Path:
        return self.workspace_dir / f"{session_id}.log"
    
    def append_log(self, session_id: str, line: str):
        """追加日志(线程安全)"""
        # 获取或创建session的日志锁
        if session_id not in self._log_locks:
            self._log_locks[session_id] = threading.Lock()
        
        with self._log_locks[session_id]:
            try:
                lp = self.get_log_path(session_id)
                with open(lp, 'a', encoding='utf-8') as f:
                    f.write(line.rstrip('\n') + '\n')
            except Exception as e:
                logger.error(f"Failed to append log: {e}")


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.storage = SessionStorage()
        self.session_types = ['Factor', 'Model', 'Kaggle', 'DataScience']
        self.statuses = ['Running', 'Completed', 'Failed', 'Stopped']
    
    def create_session(self, session_type: str, config: Dict) -> str:
        """创建新会话"""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        session = {
            'session_id': session_id,
            'type': session_type,
            'status': 'Running',
            'config': config,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'progress': 0.0,
            'current_step': None,
            'completed_steps': [],
            'error': None
        }
        
        self.storage.add_session(session)
        return session_id
    
    def list_sessions(self, session_type: Optional[str] = None, 
                     status: Optional[str] = None) -> List[Dict]:
        """列出会话（带过滤）"""
        sessions = self.storage.load_sessions()
        
        if session_type:
            sessions = [s for s in sessions if s.get('type') == session_type]
        if status:
            sessions = [s for s in sessions if s.get('status') == status]
        
        # 按创建时间降序排列
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return sessions
    
    def update_session_status(self, session_id: str, status: str, 
                            progress: float = None, error: str = None):
        """更新会话状态"""
        updates = {
            'status': status,
            'updated_at': datetime.now().isoformat()
        }
        if progress is not None:
            updates['progress'] = progress
        if error is not None:
            updates['error'] = error
        
        self.storage.update_session(session_id, updates)
    
    def stop_session(self, session_id: str):
        """停止会话"""
        try:
            self.storage.append_log(session_id, "[SYSTEM] Stop requested by user")
        except Exception:
            pass
        self.update_session_status(session_id, 'Stopped')
        # 后台任务会在下一次轮询时终止
    
    def delete_session(self, session_id: str):
        """删除会话"""
        self.storage.delete_session(session_id)
    
    def get_session_logs(self, session_id: str, tail: int = 100) -> List[str]:
        """获取会话日志"""
        return self.storage.get_session_logs(session_id, tail)
    
    # ---- 后台执行绑定 RD-Agent API ----
    def start_background_job(self, session_id: str):
        sess = self.storage.get_session(session_id)
        if not sess:
            return
        s_type = sess.get('type')
        cfg = sess.get('config', {})
        
        def worker_kaggle():
            from .rdagent_api import RDAgentAPI
            api = RDAgentAPI()
            comp = cfg.get('competition', 'titanic')
            step_n = int(cfg.get('step_n', 5))
            loop_n = int(cfg.get('loop_n', 3))
            self.storage.append_log(session_id, f"[KAGGLE] Start competition={comp}, step_n={step_n}, loop_n={loop_n}")
            try:
                for info in api.run_kaggle_rdloop_stream(comp, step_n, loop_n):
                    # 检查停止信号
                    current = self.storage.get_session(session_id)
                    if current and current.get('status') != 'Running':
                        self.storage.append_log(session_id, "[KAGGLE] Stopped by user")
                        return
                    # 更新进度
                    total = max(1, info.get('total_loops', loop_n))
                    cur = max(0, info.get('loop', 0))
                    progress = min(1.0, cur / total)
                    msg = info.get('message', '')
                    best = info.get('best_score', 0.0)
                    subs = info.get('submissions', 0)
                    self.update_session_status(session_id, 'Running', progress=progress)
                    self.storage.append_log(session_id, f"[KAGGLE] Loop {cur}/{total} submissions={subs} best={best:.5f} {msg}")
                # 完成
                self.update_session_status(session_id, 'Completed', progress=1.0)
                self.storage.append_log(session_id, "[KAGGLE] Completed")
            except Exception as e:
                self.update_session_status(session_id, 'Failed', error=str(e))
                self.storage.append_log(session_id, f"[KAGGLE][ERROR] {e}")
        
        def worker_ds():
            from .rdagent_api import RDAgentAPI
            api = RDAgentAPI()
            data_path = cfg.get('data_path')
            task_type = cfg.get('task_type', 'classification')
            step_n = int(cfg.get('step_n', 5))
            loop_n = int(cfg.get('loop_n', 1))
            timeout = cfg.get('timeout')  # None or int
            metric = cfg.get('metric', 'auto')
            
            # 构建日志消息
            log_msg = f"[DS] Start data_path={data_path}, task={task_type}, step_n={step_n}, loop_n={loop_n}"
            if timeout:
                log_msg += f", timeout={timeout}s"
            self.storage.append_log(session_id, log_msg)
            
            try:
                result = api.run_data_science(data_path, {
                    'task_type': task_type,
                    'metric': metric,
                    'step_n': step_n,
                    'loop_n': loop_n,
                    'timeout': timeout,
                })
                if result.get('success'):
                    score = result.get('score')
                    self.storage.append_log(session_id, f"[DS] Done metric={result.get('metric')} score={score}")
                    self.update_session_status(session_id, 'Completed', progress=1.0)
                else:
                    self.storage.append_log(session_id, f"[DS][ERROR] {result.get('message')}")
                    self.update_session_status(session_id, 'Failed', error=result.get('message'))
            except Exception as e:
                self.update_session_status(session_id, 'Failed', error=str(e))
                self.storage.append_log(session_id, f"[DS][ERROR] {e}")
        
        t = None
        if s_type == 'Kaggle':
            t = threading.Thread(target=worker_kaggle, daemon=True)
        elif s_type == 'DataScience':
            t = threading.Thread(target=worker_ds, daemon=True)
        else:
            # 其他类型可按需扩展
            return
        t.start()


def render_session_list(manager: SessionManager):
    """渲染会话列表"""
    st.subheader("📋 会话列表")
    
    # 自动刷新控制
    col_refresh1, col_refresh2 = st.columns([1, 3])
    with col_refresh1:
        auto_refresh = st.checkbox("✅ 自动刷新", value=False, key="session_auto_refresh")
    with col_refresh2:
        if auto_refresh:
            refresh_interval = st.slider("刷新间隔(秒)", 1, 10, 3, key="session_refresh_interval")
            st.caption(f"将每 {refresh_interval} 秒自动刷新一次")
            # Streamlit 自动刷新机制
            import time
            time.sleep(refresh_interval)
            st.rerun()
    
    # 过滤器
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox(
            "会话类型",
            options=['全部'] + manager.session_types,
            key="session_type_filter"
        )
    with col2:
        status_filter = st.selectbox(
            "状态",
            options=['全部'] + manager.statuses,
            key="session_status_filter"
        )
    with col3:
        if st.button("🔄 手动刷新", key="refresh_sessions"):
            st.rerun()
    
    # 获取会话列表
    type_f = type_filter if type_filter != '全部' else None
    status_f = status_filter if status_filter != '全部' else None
    sessions = manager.list_sessions(session_type=type_f, status=status_f)
    
    if not sessions:
        st.info("暂无会话记录")
        return
    
    # 显示会话表格
    import pandas as pd
    df_data = []
    for sess in sessions:
        df_data.append({
            '会话ID': sess['session_id'],
            '类型': sess['type'],
            '状态': sess['status'],
            '进度': f"{sess.get('progress', 0):.0%}",
            '创建时间': sess['created_at'][:19],
            '更新时间': sess.get('updated_at', '')[:19]
        })
    
    df = pd.DataFrame(df_data)
    
    # 使用颜色标记状态
    def highlight_status(row):
        if row['状态'] == 'Running':
            return ['background-color: #d4edda'] * len(row)
        elif row['状态'] == 'Completed':
            return ['background-color: #d1ecf1'] * len(row)
        elif row['状态'] == 'Failed':
            return ['background-color: #f8d7da'] * len(row)
        else:
            return [''] * len(row)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 会话详情和操作
    if sessions:
        st.subheader("🔍 会话详情")
        selected_id = st.selectbox(
            "选择会话",
            options=[s['session_id'] for s in sessions],
            format_func=lambda x: f"{x} - {next((s['type'] for s in sessions if s['session_id']==x), '')}",
            key="selected_session_id"
        )
        
        if selected_id:
            render_session_detail(manager, selected_id)


def render_session_detail(manager: SessionManager, session_id: str):
    """渲染会话详情"""
    session = manager.storage.get_session(session_id)
    if not session:
        st.error(f"会话 {session_id} 不存在")
        return
    
    # 基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("类型", session['type'])
    with col2:
        status = session['status']
        status_emoji = {
            'Running': '🟢',
            'Completed': '✅',
            'Failed': '❌',
            'Stopped': '⏸️'
        }
        st.metric("状态", f"{status_emoji.get(status, '')} {status}")
    with col3:
        st.metric("进度", f"{session.get('progress', 0):.0%}")
    with col4:
        created = datetime.fromisoformat(session['created_at'])
        duration = (datetime.now() - created).total_seconds()
        st.metric("运行时长", f"{duration/60:.1f}分钟")
    
    # 进度条
    st.progress(session.get('progress', 0.0))
    
    # 配置信息
    with st.expander("⚙️ 配置信息"):
        st.json(session.get('config', {}))
    
    # RD-Agent 日志路径指引
    with st.expander("📂 RD-Agent 日志路径"):
        st.caption("查看底层 RD-Agent 详细日志")
        
        # 获取可能的日志路径
        import os
        from pathlib import Path
        
        log_paths = []
        
        # 优先级1: 环境变量 RDAGENT_LOG_PATH
        rdagent_log_env = os.getenv('RDAGENT_LOG_PATH')
        if rdagent_log_env:
            log_path_1 = Path(rdagent_log_env)
            log_paths.append(("环境变量 RDAGENT_LOG_PATH", str(log_path_1), log_path_1.exists()))
        
        # 优先级2: ~/.rdagent/log (官方默认)
        home_log = Path.home() / '.rdagent' / 'log'
        log_paths.append(("用户目录 (官方默认)", str(home_log), home_log.exists()))
        
        # 优先级3: $RDAGENT_PATH/log
        rdagent_path = os.getenv('RDAGENT_PATH')
        if rdagent_path:
            rdagent_log = Path(rdagent_path) / 'log'
            log_paths.append(("RDAGENT_PATH/log", str(rdagent_log), rdagent_log.exists()))
        
        # 优先级4: ./workspace/log
        workspace_log = Path.cwd() / 'workspace' / 'log'
        log_paths.append(("工作目录", str(workspace_log), workspace_log.exists()))
        
        # 显示日志路径
        found_active = False
        for label, path_str, exists in log_paths:
            if exists:
                st.success(f"✅ **{label}**: `{path_str}`")
                
                # 添加复制按钮（仅第一个存在的路径）
                if not found_active:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.caption("💡 可以在文件管理器中打开此路径查看详细日志")
                    with col_b:
                        # 使用 st.code 让用户可以直接复制
                        pass
                    found_active = True
            else:
                st.info(f"⚪ {label}: `{path_str}` (不存在)")
        
        if not found_active:
            st.warning("未找到 RD-Agent 日志目录，可能尚未运行过任何任务")
        else:
            st.caption("ℹ️ 日志目录优先级：RDAGENT_LOG_PATH > ~/.rdagent/log > $RDAGENT_PATH/log > ./workspace/log")
    
    # 错误信息
    if session.get('error'):
        with st.expander("❌ 错误信息", expanded=True):
            st.error(session['error'])
    
    # 操作按钮
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if session['status'] == 'Running':
            if st.button("⏸️ 停止", key=f"stop_{session_id}"):
                manager.stop_session(session_id)
                st.success("会话已停止")
                st.rerun()
    with col2:
        if st.button("🗑️ 删除", key=f"delete_{session_id}"):
            manager.delete_session(session_id)
            st.success("会话已删除")
            st.rerun()
    with col3:
        if st.button("📥 导出日志", key=f"export_{session_id}"):
            logs = manager.get_session_logs(session_id, tail=0)
            log_text = ''.join(logs)
            st.download_button(
                label="下载日志",
                data=log_text,
                file_name=f"session_{session_id}.log",
                mime="text/plain"
            )
    
    # 日志查看
    st.subheader("📜 日志")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tail_lines = st.number_input(
            "显示最后N行",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key=f"tail_{session_id}"
        )
    with col2:
        log_filter = st.text_input("过滤关键词", key=f"filter_{session_id}")
    with col3:
        if st.button("🔄 刷新日志", key=f"refresh_log_{session_id}"):
            st.rerun()
    
    # 获取日志
    logs = manager.get_session_logs(session_id, tail=int(tail_lines))
    
    # 过滤日志
    if log_filter:
        logs = [line for line in logs if log_filter.lower() in line.lower()]
    
    if logs:
        log_text = ''.join(logs)
        st.code(log_text, language='log', line_numbers=True)
    else:
        st.info("暂无日志")


def render_new_session_form(manager: SessionManager):
    """渲染创建新会话表单"""
    st.subheader("➕ 创建新会话")
    
    st.info("💡 提示：所有参数已预设合理默认值，直接点击启动即可使用。也可根据需要调整参数。")
    
    session_type = st.selectbox(
        "会话类型",
        options=manager.session_types,
        key="new_session_type",
        help="选择你想运行的任务类型"
    )
    
    st.write("**配置参数**")
    st.caption("⭐ 以下参数已设为推荐值，新手可直接使用")
    
    if session_type == 'Factor':
        max_factors = st.number_input(
            "最大因子数", 
            min_value=1, max_value=20, value=5,
            help="生成因子的数量，推荐 5 个"
        )
        factor_type = st.selectbox(
            "因子类型", 
            ['技术因子', '基本面因子', '情绪因子'],
            help="技术因子基于价格/成交量，基本面因子基于财务数据，情绪因子基于市场情绪"
        )
        config = {
            'max_factors': max_factors,
            'factor_type': factor_type
        }
    elif session_type == 'Model':
        max_trials = st.number_input(
            "最大试验次数", 
            min_value=1, max_value=50, value=10,
            help="超参搜索的次数，推荐 10 次"
        )
        search_method = st.selectbox(
            "搜索方法", 
            ['Random', 'Grid', 'Bayesian'],
            index=2,
            help="Bayesian 贝叶斯优化通常最快，推荐新手使用"
        )
        config = {
            'max_trials': max_trials,
            'search_method': search_method
        }
    elif session_type == 'Kaggle':
        st.caption("🏆 Kaggle 竞赛自动化：系统将自动下载数据、生成模型、提交结果")
        competition = st.text_input(
            "竞赛名称", 
            "titanic",
            help="推荐新手使用 titanic 或 house-prices-advanced-regression-techniques"
        )
        step_n = st.number_input(
            "每轮步数 step_n", 
            min_value=1, max_value=20, value=3,
            help="每轮循环的迭代步数，推荐 3-5"
        )
        loop_n = st.number_input(
            "循环次数 loop_n", 
            min_value=1, max_value=20, value=2,
            help="总共运行几轮，推荐 2-3 轮（总耗时 = step_n × loop_n）"
        )
        config = {
            'competition': competition,
            'step_n': step_n,
            'loop_n': loop_n
        }
    else:  # DataScience
        st.caption("🧪 数据科学自动建模：上传数据，系统自动分析、特征工程、模型训练")
        task_type = st.selectbox(
            "任务类型", 
            ['classification', 'regression'],
            help="分类任务（预测类别）或回归任务（预测数值）"
        )
        # 循环控制参数
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            step_n = st.number_input(
                "步数 step_n", 
                min_value=1, max_value=50, value=5,
                help="每轮运行的步数，推荐 5"
            )
        with col_ds2:
            loop_n = st.number_input(
                "循环次数 loop_n", 
                min_value=1, max_value=20, value=1,
                help="循环运行的次数，推荐 1-2 轮"
            )
        with col_ds3:
            timeout = st.number_input(
                "超时(秒)", 
                min_value=0, max_value=7200, value=0,
                help="总运行时长限制，0=不限制。推荐 1800（30分钟）"
            )
        # 生成默认示例数据路径
        default_data_dir = Path.cwd() / "workspace" / "data_science" / "sample_data"
        default_data_dir.mkdir(parents=True, exist_ok=True)
        default_csv = default_data_dir / "sample_train.csv"
        
        # 如果示例数据不存在，自动生成
        if not default_csv.exists():
            try:
                import pandas as pd
                import numpy as np
                # 生成简单分类示例数据
                np.random.seed(42)
                n = 200
                df = pd.DataFrame({
                    'feature_1': np.random.randn(n),
                    'feature_2': np.random.randn(n),
                    'feature_3': np.random.rand(n) * 100,
                    'feature_4': np.random.choice(['A', 'B', 'C'], n),
                    'target': np.random.choice([0, 1], n)
                })
                df.to_csv(default_csv, index=False)
            except Exception:
                pass
        
        # 显示示例数据说明
        if default_csv.exists():
            st.success(f"✅ 已自动生成示例数据: `{default_csv.name}` (200行×5列分类数据)")
            with st.expander("👁️ 查看示例数据路径"):
                st.code(str(default_csv), language="text")
        
        data_path = st.text_input(
            "数据路径", 
            value=str(default_csv) if default_csv.exists() else "",
            placeholder="例如: C:/Users/YourName/Documents/my_data.csv",
            help="默认使用示例数据。若要使用自己的数据，请输入完整路径（支持 CSV/Excel/Parquet）"
        )
        
        # 真实数据路径示例提示
        st.caption("📝 真实数据路径示例：`G:\\\\data\\\\train.csv` 或 `C:\\\\Users\\\\Administrator\\\\Documents\\\\data.xlsx`")
        
        # 数据准备指南
        with st.expander("📚 如何准备自己的数据？（新手必看）"):
            st.markdown("""
            ### 📊 数据准备指南
            
            #### ① 数据格式要求
            - **支持格式**: CSV (.csv) / Excel (.xlsx) / Parquet (.parquet)
            - **文件结构**: 表格数据，每列为一个特征，最后一列为目标值
            - **示例结构**:
            
            | feature_1 | feature_2 | feature_3 | target |
            |-----------|-----------|-----------|--------|
            | 1.5       | 20        | A         | 0      |
            | 2.3       | 35        | B         | 1      |
            | 0.8       | 18        | A         | 0      |
            
            ---
            
            #### ② 快速获取数据路径（Windows）
            
            **方法 1：直接拖拽**
            1. 在文件资源管理器中找到你的数据文件
            2. **按住 Shift 键** + **右键点击文件**
            3. 选择「**复制为路径**」
            4. 粘贴到上方输入框
            
            **方法 2：查看属性**
            1. 右键点击文件 → 「属性」
            2. 复制「位置」栏的路径
            3. 手动添加文件名，例如：
               - 位置: `C:\\Users\\Administrator\\Documents`
               - 文件名: `my_data.csv`
               - **完整路径**: `C:\\Users\\Administrator\\Documents\\my_data.csv`
            
            ---
            
            #### ③ 建议的数据存放位置
            - **桌面**: `C:\\Users\\Administrator\\Desktop\\my_data.csv`
            - **文档**: `C:\\Users\\Administrator\\Documents\\data\\train.csv`
            - **项目目录**: `G:\\test\\qilin_stack\\data\\my_dataset.csv`
            
            ---
            
            #### ④ 注意事项
            - ⚠️ **路径不能包含中文**（建议用英文文件夹名）
            - ⚠️ **不要有空格**（或用引号包裹）
            - ✅ **推荐路径**: `G:\\data\\train.csv`
            - ❌ **避免路径**: `G:\\我的 文件夹\\数据.csv`
            
            ---
            
            #### ⑤ 快速测试：使用示例数据
            如果没有自己的数据，**直接使用默认示例数据即可**！
            系统已自动生成 200 行测试数据，可以直接点击「一键启动」体验功能。
            
            ---
            
            #### ⑥ 还是不会？
            👉 **最简单方法**：
            1. 在桌面创建一个 Excel 文件
            2. 填入一些数据（表头 + 几行数据）
            3. 另存为 `.csv` 格式
            4. 右键点击文件 → 属性 → 复制完整路径
            5. 粘贴到上方输入框
            """)
        metric = st.text_input(
            "评估指标(可选)", 
            value="auto",
            help="留空或填 auto 则自动选择，也可手动指定如 accuracy/f1/rmse"
        )
        config = {
            'task_type': task_type,
            'step_n': step_n,
            'loop_n': loop_n,
            'timeout': timeout if timeout > 0 else None,
            'data_path': data_path,
            'metric': metric,
        }
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        if st.button("🚀 一键启动", type="primary", use_container_width=True):
            session_id = manager.create_session(session_type, config)
            manager.start_background_job(session_id)
            st.success(f"✅ 会话已创建: {session_id}")
            st.balloons()
            st.info("🔄 会话正在后台运行，请在「会话列表」页查看实时进度与日志")
            time.sleep(1)
            st.rerun()
    with col_btn2:
        st.caption("👉 点击后系统将自动开始工作，无需额外操作")


def render():
    """主渲染函数"""
    st.title("🎮 RD-Agent 会话管理")
    st.caption("管理和监控RD-Agent的运行会话")
    
    # 初始化管理器
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    
    manager = st.session_state.session_manager
    
    # 主标签页
    tab1, tab2 = st.tabs(["📋 会话列表", "➕ 创建会话"])
    
    with tab1:
        render_session_list(manager)
    
    with tab2:
        render_new_session_form(manager)


if __name__ == "__main__":
    render()
