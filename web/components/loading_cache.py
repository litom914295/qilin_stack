"""
加载动画和数据缓存管理器
提供优雅的加载提示和智能数据缓存
"""

import streamlit as st
import time
import functools
from typing import Any, Callable, Optional
from datetime import datetime, timedelta
from .color_scheme import Colors, Emojis


# ==================== 加载动画组件 ====================

class LoadingSpinner:
    """加载动画上下文管理器"""
    
    def __init__(self, message: str = "加载中...", emoji: str = "⏳"):
        self.message = message
        self.emoji = emoji
        self.placeholder = None
    
    def __enter__(self):
        """进入上下文时显示加载动画"""
        self.placeholder = st.empty()
        self.show_loading()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时清除加载动画"""
        if self.placeholder:
            self.placeholder.empty()
        return False
    
    def show_loading(self):
        """显示加载动画"""
        if self.placeholder:
            html = f"""
            <div class="fade-in" style="
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                gap: 12px;
            ">
                <div class="spinner"></div>
                <span style="font-size: 16px; color: {Colors.TEXT_MUTED};">
                    {self.emoji} {self.message}
                </span>
            </div>
            """
            self.placeholder.markdown(html, unsafe_allow_html=True)
    
    def update_message(self, message: str, emoji: str = "⏳"):
        """更新加载消息"""
        self.message = message
        self.emoji = emoji
        self.show_loading()


def show_progress_bar(
    current: int,
    total: int,
    message: str = "处理中",
    color: str = Colors.PRIMARY
):
    """
    显示进度条
    
    Args:
        current: 当前进度
        total: 总数
        message: 提示消息
        color: 进度条颜色
    """
    percentage = int((current / total) * 100) if total > 0 else 0
    
    html = f"""
    <div class="fade-in" style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 14px; color: {Colors.TEXT_DARK};">{message}</span>
            <span style="font-size: 14px; font-weight: 600; color: {color};">{current}/{total} ({percentage}%)</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%; background-color: {color};"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_skeleton_loader(rows: int = 3):
    """
    显示骨架屏加载动画
    
    Args:
        rows: 骨架屏行数
    """
    html = """
    <style>
        @keyframes skeleton-loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        .skeleton-line {
            height: 16px;
            margin: 12px 0;
            background: linear-gradient(
                90deg,
                #f0f0f0 25%,
                #e0e0e0 50%,
                #f0f0f0 75%
            );
            background-size: 200% 100%;
            animation: skeleton-loading 1.5s infinite;
            border-radius: 4px;
        }
    </style>
    """
    
    for i in range(rows):
        width = 100 if i < rows - 1 else 60
        html += f'<div class="skeleton-line" style="width: {width}%;"></div>'
    
    st.markdown(html, unsafe_allow_html=True)


def show_success_animation(message: str = "操作成功", duration: float = 2.0):
    """
    显示成功动画
    
    Args:
        message: 成功消息
        duration: 显示时长（秒）
    """
    placeholder = st.empty()
    
    html = f"""
    <div class="fade-in" style="
        background-color: {Colors.BG_SUCCESS};
        border: 2px solid {Colors.SUCCESS};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: {Colors.SUCCESS};
    ">
        {Emojis.SUCCESS} {message}
    </div>
    """
    
    placeholder.markdown(html, unsafe_allow_html=True)
    time.sleep(duration)
    placeholder.empty()


def show_error_animation(message: str = "操作失败", duration: float = 3.0):
    """
    显示错误动画
    
    Args:
        message: 错误消息
        duration: 显示时长（秒）
    """
    placeholder = st.empty()
    
    html = f"""
    <div class="fade-in" style="
        background-color: {Colors.BG_DANGER};
        border: 2px solid {Colors.DANGER};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: {Colors.DANGER};
    ">
        {Emojis.ERROR} {message}
    </div>
    """
    
    placeholder.markdown(html, unsafe_allow_html=True)
    time.sleep(duration)
    placeholder.empty()


# ==================== 数据缓存管理 ====================

class CacheManager:
    """缓存管理器"""
    
    @staticmethod
    def cache_data(ttl: int = 300, show_spinner: bool = True):
        """
        数据缓存装饰器（使用Streamlit的cache_data）
        
        Args:
            ttl: 缓存生存时间（秒），默认5分钟
            show_spinner: 是否显示加载提示
            
        Returns:
            装饰后的函数
        """
        return st.cache_data(ttl=ttl, show_spinner=show_spinner)
    
    @staticmethod
    def cache_resource(show_spinner: bool = True):
        """
        资源缓存装饰器（使用Streamlit的cache_resource）
        适用于数据库连接、模型加载等单例资源
        
        Args:
            show_spinner: 是否显示加载提示
            
        Returns:
            装饰后的函数
        """
        return st.cache_resource(show_spinner=show_spinner)
    
    @staticmethod
    def clear_cache():
        """清除所有缓存"""
        st.cache_data.clear()
        st.cache_resource.clear()
    
    @staticmethod
    def get_cache_stats():
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        # Streamlit不直接提供缓存统计，这里返回占位信息
        return {
            "total_cached_functions": "N/A",
            "cache_size": "N/A",
            "last_cleared": "N/A"
        }


def cached_query(ttl: int = 300):
    """
    自定义缓存装饰器，带有详细的加载提示
    
    Args:
        ttl: 缓存生存时间（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        # 使用functools.wraps保留原函数信息
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # 检查session_state中的缓存
            if 'query_cache' not in st.session_state:
                st.session_state.query_cache = {}
            
            cache_info = st.session_state.query_cache.get(cache_key)
            
            # 检查缓存是否有效
            if cache_info:
                cached_time, cached_data = cache_info
                if datetime.now() - cached_time < timedelta(seconds=ttl):
                    # 缓存命中
                    return cached_data
            
            # 缓存未命中，执行查询
            with LoadingSpinner(f"正在执行 {func.__name__}...", Emojis.REFRESH):
                result = func(*args, **kwargs)
            
            # 保存到缓存
            st.session_state.query_cache[cache_key] = (datetime.now(), result)
            
            return result
        
        return wrapper
    return decorator


# ==================== 性能监控 ====================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, operation_name: str, show_result: bool = True):
        self.operation_name = operation_name
        self.show_result = show_result
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """结束计时并显示结果"""
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        
        if self.show_result:
            if elapsed_time < 1:
                time_str = f"{elapsed_time * 1000:.0f}ms"
                color = Colors.SUCCESS
            elif elapsed_time < 3:
                time_str = f"{elapsed_time:.2f}s"
                color = Colors.WARNING
            else:
                time_str = f"{elapsed_time:.2f}s"
                color = Colors.DANGER
            
            st.caption(f"⏱️ {self.operation_name} 耗时: {time_str}")
        
        return False
    
    def get_elapsed_time(self) -> float:
        """获取已用时间"""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        else:
            return 0.0


def time_function(func: Callable) -> Callable:
    """
    函数执行时间装饰器
    
    Args:
        func: 要监控的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceMonitor(func.__name__, show_result=True):
            return func(*args, **kwargs)
    return wrapper


# ==================== 批量操作优化 ====================

def batch_process(
    items: list,
    process_func: Callable,
    batch_size: int = 100,
    show_progress: bool = True
) -> list:
    """
    批量处理数据，带进度显示
    
    Args:
        items: 待处理的项目列表
        process_func: 处理函数
        batch_size: 批次大小
        show_progress: 是否显示进度
        
    Returns:
        处理结果列表
    """
    results = []
    total = len(items)
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
        
        if show_progress:
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"处理进度: {min(i + batch_size, total)}/{total}")
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    return results


# ==================== 懒加载支持 ====================

class LazyLoader:
    """懒加载器 - 仅在需要时加载数据"""
    
    def __init__(self, load_func: Callable, *args, **kwargs):
        self.load_func = load_func
        self.args = args
        self.kwargs = kwargs
        self._data = None
        self._loaded = False
    
    def load(self):
        """加载数据"""
        if not self._loaded:
            with LoadingSpinner("正在加载数据...", Emojis.CHART):
                self._data = self.load_func(*self.args, **self.kwargs)
                self._loaded = True
        return self._data
    
    @property
    def data(self):
        """获取数据（自动加载）"""
        return self.load()
    
    def is_loaded(self) -> bool:
        """检查是否已加载"""
        return self._loaded
    
    def reload(self):
        """重新加载数据"""
        self._loaded = False
        return self.load()


# ==================== 导出 ====================

__all__ = [
    'LoadingSpinner',
    'show_progress_bar',
    'show_skeleton_loader',
    'show_success_animation',
    'show_error_animation',
    'CacheManager',
    'cached_query',
    'PerformanceMonitor',
    'time_function',
    'batch_process',
    'LazyLoader',
]
