"""
键盘快捷键管理器
为Streamlit应用添加键盘快捷键支持
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Callable, Optional


class KeyboardShortcuts:
    """键盘快捷键管理器"""
    
    # 默认快捷键映射
    DEFAULT_SHORTCUTS = {
        'r': '刷新数据',
        'e': '导出报告',
        's': '保存候选池',
        'f': '筛选数据',
        'h': '显示帮助',
        '1': '切换到T日选股',
        '2': '切换到T+1竞价',
        '3': '切换到T+2卖出',
        '4': '切换到统计分析',
        'Escape': '关闭弹窗',
    }
    
    def __init__(self):
        self.callbacks: Dict[str, Callable] = {}
        self.enabled = True
    
    def register(self, key: str, callback: Callable, description: str = ""):
        """
        注册快捷键
        
        Args:
            key: 快捷键（如'r', 'ctrl+s'）
            callback: 回调函数
            description: 快捷键描述
        """
        self.callbacks[key.lower()] = {
            'callback': callback,
            'description': description or self.DEFAULT_SHORTCUTS.get(key, "")
        }
    
    def enable(self):
        """启用快捷键"""
        self.enabled = True
    
    def disable(self):
        """禁用快捷键"""
        self.enabled = False
    
    def inject_js(self):
        """注入JavaScript监听快捷键"""
        
        if not self.enabled:
            return
        
        # 生成键位到session_state key的映射
        key_mapping = {
            key: f"shortcut_pressed_{key}"
            for key in self.callbacks.keys()
        }
        
        js_code = """
        <script>
        // 键盘快捷键监听器
        (function() {
            // 防止重复注册
            if (window.keyboardShortcutsInitialized) {
                return;
            }
            window.keyboardShortcutsInitialized = true;
            
            // 快捷键映射
            const shortcuts = %s;
            
            // 监听键盘事件
            document.addEventListener('keydown', function(e) {
                // 排除输入框内的快捷键
                const tagName = e.target.tagName.toLowerCase();
                if (tagName === 'input' || tagName === 'textarea') {
                    return;
                }
                
                const key = e.key.toLowerCase();
                
                // 检查是否是注册的快捷键
                if (shortcuts[key]) {
                    e.preventDefault();
                    
                    // 触发Streamlit重新运行
                    const sessionState = shortcuts[key];
                    
                    // 使用Streamlit的setComponentValue触发更新
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        key: sessionState,
                        value: Date.now()
                    }, '*');
                    
                    console.log('快捷键触发:', key);
                }
            });
            
            console.log('键盘快捷键已初始化');
        })();
        </script>
        """ % str(key_mapping).replace("'", '"')
        
        components.html(js_code, height=0)
    
    def check_pressed(self, key: str) -> bool:
        """
        检查快捷键是否被按下
        
        Args:
            key: 快捷键
            
        Returns:
            是否被按下
        """
        state_key = f"shortcut_pressed_{key}"
        
        if state_key in st.session_state:
            # 清除状态
            del st.session_state[state_key]
            return True
        
        return False
    
    def show_help(self):
        """显示快捷键帮助"""
        help_text = "## ⌨️ 键盘快捷键\n\n"
        
        for key, info in self.callbacks.items():
            description = info.get('description', '')
            if description:
                help_text += f"- **{key.upper()}**: {description}\n"
        
        st.info(help_text)


# 全局快捷键实例
_global_shortcuts = None


def get_keyboard_shortcuts() -> KeyboardShortcuts:
    """获取全局快捷键实例"""
    global _global_shortcuts
    if _global_shortcuts is None:
        _global_shortcuts = KeyboardShortcuts()
    return _global_shortcuts


def setup_default_shortcuts():
    """
    设置默认快捷键
    返回快捷键管理器实例
    """
    shortcuts = get_keyboard_shortcuts()
    
    # 注册默认快捷键（只注册描述，实际回调由具体页面处理）
    for key, desc in KeyboardShortcuts.DEFAULT_SHORTCUTS.items():
        shortcuts.register(key, lambda: None, desc)
    
    # 注入JavaScript
    shortcuts.inject_js()
    
    return shortcuts


__all__ = [
    'KeyboardShortcuts',
    'get_keyboard_shortcuts',
    'setup_default_shortcuts',
]
