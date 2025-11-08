"""
RD-Agent 环境与运行配置
- RDAGENT_PATH 与依赖健康检查
- .env 管理（关键变量：DS_LOCAL_DATA_PATH、DS_IF_USING_MLE_DATA、DS_CODER_COSTEER_ENV_TYPE）
- Kaggle API / Docker / Kaggle CLI 检查
- 一键诊断与修复
"""

import os
from pathlib import Path
from typing import Dict, Any
import streamlit as st


ENV_FILE = Path('.env')


def load_dotenv_file() -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if ENV_FILE.exists():
        try:
            for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                vals[k.strip()] = v.strip()
        except Exception:
            pass
    return vals


def save_dotenv_file(kv: Dict[str, str]) -> None:
    # 合并已有项
    existing = load_dotenv_file()
    existing.update({k: str(v) for k, v in kv.items() if v is not None})
    # 按键排序写回
    lines = [f"{k}={existing[k]}" for k in sorted(existing.keys())]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding='utf-8')


def render_env_summary(status: Dict[str, Any]):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RD-Agent可用", '✅' if status.get('rdagent_importable') else '❌')
    with col2:
        st.metric("Kaggle API", '✅' if status.get('kaggle_api_configured') else '❌')
    with col3:
        st.metric("Kaggle CLI", '✅' if status.get('kaggle_cli') else '❌')
    with col4:
        st.metric("Docker", '✅' if status.get('docker') else '❌')


def render_env_editor():
    st.subheader(".env 关键参数")
    env_vals = load_dotenv_file()

    col1, col2 = st.columns(2)
    with col1:
        rdagent_path = st.text_input(
            "RDAGENT_PATH (RD-Agent源码/包目录)",
            value=os.getenv('RDAGENT_PATH', env_vals.get('RDAGENT_PATH', '')),
            help="用于从源码导入 rdagent；留空则按已安装包解析"
        )
    with col2:
        ds_local_data = st.text_input(
            "DS_LOCAL_DATA_PATH (数据根目录)",
            value=os.getenv('DS_LOCAL_DATA_PATH', env_vals.get('DS_LOCAL_DATA_PATH', str(Path('data/ds_data')))),
            help="Data Science/Kaggle 场景的数据目录"
        )
    col3, col4 = st.columns(2)
    with col3:
        ds_using_mle = st.checkbox(
            "DS_IF_USING_MLE_DATA (使用MLE‑Bench数据流水线)",
            value=(os.getenv('DS_IF_USING_MLE_DATA', env_vals.get('DS_IF_USING_MLE_DATA', 'False')).lower() == 'true')
        )
    with col4:
        env_type = st.selectbox(
            "DS_CODER_COSTEER_ENV_TYPE (运行环境)",
            options=["docker", "conda"],
            index=["docker", "conda"].index(os.getenv('DS_CODER_COSTEER_ENV_TYPE', env_vals.get('DS_CODER_COSTEER_ENV_TYPE', 'conda'))),
            help="Windows系统推荐使用conda，Linux/Mac可选docker"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💾 保存到 .env", type="primary"):
            save_dotenv_file({
                'RDAGENT_PATH': rdagent_path,
                'DS_LOCAL_DATA_PATH': ds_local_data,
                'DS_IF_USING_MLE_DATA': str(bool(ds_using_mle)),
                'DS_CODER_COSTEER_ENV_TYPE': env_type,
            })
            st.success("已写入 .env，重启应用后生效（或刷新并确保已加载环境变量）")
    with c2:
        if st.button("♻️ 刷新状态"):
            st.rerun()
    with c3:
        st.caption(ENV_FILE.absolute())


def run_health_check() -> Dict[str, Any]:
    # 通过 RDAgentAPI 统一检查
    try:
        from .rdagent_api import RDAgentAPI
        api = RDAgentAPI()
        return api.health_check()
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'rdagent_importable': False,
            'details': {}
        }


def render_diagnostics(status: Dict[str, Any]):
    st.subheader("🩺 一键诊断")

    with st.expander("检查详情", expanded=True):
        st.json(status)

    fixes = []
    if not status.get('kaggle_api_configured'):
        fixes.append("未检测到 ~/.kaggle/kaggle.json，请在 Kaggle 账户设置生成并放置文件；Windows 建议放置到 %USERPROFILE%/.kaggle/kaggle.json")
    if not status.get('kaggle_cli'):
        fixes.append("未检测到 Kaggle CLI，建议安装：pip install kaggle，并将命令加入 PATH")
    if not status.get('docker') and status.get('env_type') == 'docker':
        fixes.append("当前选择 docker 运行，但未检测到 Docker，请安装 Docker Desktop 并确保可用")
    if not status.get('rdagent_importable'):
        fixes.append("无法导入 rdagent：确保已 pip install rdagent 或设置 RDAGENT_PATH 指向源码路径")

    if fixes:
        st.warning("发现以下待处理项：")
        for f in fixes:
            st.write(f"- {f}")
    else:
        st.success("环境检查通过 ✅")


def render():
    st.title("⚙️ RD‑Agent 环境与运行配置")

    st.caption("配置 RD‑Agent 路径、数据目录与运行环境，并进行一键诊断。")
    
    # 日志根目录说明
    with st.expander("📁 日志根目录优先级说明", expanded=False):
        st.markdown("""
        ### 📂 RD-Agent 日志目录优先级
        
        系统按以下优先级自动查找日志：
        
        1. **~/.rdagent/log** （RD-Agent 默认日志目录）
        2. **$RDAGENT_PATH/log** （若配置了 RDAGENT_PATH）
        3. **./workspace/log** （项目本地工作空间）
        
        ---
        
        ### 🔧 自定义日志目录
        
        可通过环境变量 **RDAGENT_LOG_ROOT** 覆盖默认路径：
        
        ```bash
        # Windows PowerShell
        $env:RDAGENT_LOG_ROOT = "G:\\my_logs"
        
        # Linux/Mac
        export RDAGENT_LOG_ROOT="/path/to/logs"
        ```
        
        ---
        
        ### 📊 日志类型
        
        - **pkl 文件**: 完整的结构化日志（包含 LLM 调用、token 成本、阶段信息）
        - **trace.json**: 简化的追踪日志（兜底方案）
        
        日志可视化工具优先使用 pkl 文件，若不可用则回退到 trace.json。
        """)

    # 1) 运行健康检查
    status = run_health_check()
    render_env_summary(status)
    st.divider()

    # 2) .env 编辑
    render_env_editor()

    st.divider()
    # 3) 诊断详情与修复建议
    render_diagnostics(status)


def main():
    render()


if __name__ == "__main__":
    main()
