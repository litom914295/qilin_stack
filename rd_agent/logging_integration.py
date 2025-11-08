"""
麒麟 RD-Agent 日志集成模块

集成官方 RD-Agent 的 FileStorage 日志系统,支持:
- pkl 格式存储实验对象
- json 格式存储指标数据
- 离线读取历史实验
- 兼容官方日志格式

作者: AI Agent
日期: 2024
"""

from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime
import logging

try:
    from rdagent.log.storage import FileStorage
    from rdagent.log.base import Message
    FILESTORAGE_AVAILABLE = True
except ImportError:
    FILESTORAGE_AVAILABLE = False
    FileStorage = None
    Message = None

logger = logging.getLogger(__name__)


class QilinRDAgentLogger:
    """
    麒麟 RD-Agent 日志管理器
    
    功能:
    1. 集成官方 FileStorage (pkl/json 格式)
    2. 支持离线读取历史实验
    3. 兼容官方日志目录结构
    4. 提供多级兜底策略
    
    使用示例:
        logger = QilinRDAgentLogger('./logs/rdagent')
        
        # 记录实验
        logger.log_experiment(exp, tag='limitup.factor')
        
        # 记录指标
        logger.log_metrics({'ic': 0.05, 'ir': 0.8})
        
        # 读取历史实验
        for exp in logger.iter_experiments():
            print(exp.hypothesis)
    """
    
    def __init__(self, workspace_path: str):
        """
        初始化日志管理器
        
        Args:
            workspace_path: 工作目录路径 (存储日志文件)
        
        Raises:
            ImportError: 如果官方 RD-Agent 不可用
        """
        if not FILESTORAGE_AVAILABLE:
            raise ImportError(
                "Official RD-Agent FileStorage not available. "
                "Please ensure rdagent is installed: pip install rdagent"
            )
        
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # 创建官方 FileStorage
        self.storage = FileStorage(self.workspace_path)
        
        # 标准 logging 作为备份
        self.logger = logging.getLogger("qilin.rd_agent")
        
        logger.info(f"QilinRDAgentLogger initialized at {self.workspace_path}")
    
    def log_experiment(self, exp: Any, tag: str = "limitup.factor") -> Path:
        """
        记录实验对象 (pkl 格式)
        
        Args:
            exp: 实验对象 (通常是 Experiment 类型)
            tag: 标签 (用于组织目录结构,例如 'limitup.factor')
        
        Returns:
            保存的文件路径
        
        Example:
            path = logger.log_experiment(exp, tag='limitup.factor')
            print(f'Saved to {path}')
        """
        try:
            path = self.storage.log(exp, tag=tag, save_type="pkl")
            self.logger.info(f"✅ Logged experiment to {tag} -> {path}")
            return path
        except Exception as e:
            self.logger.error(f"❌ Failed to log experiment: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Any], tag: str = "limitup.metrics") -> Path:
        """
        记录指标数据 (json 格式)
        
        Args:
            metrics: 指标字典 (例如 {'ic': 0.05, 'ir': 0.8})
            tag: 标签 (用于组织目录结构)
        
        Returns:
            保存的文件路径
        
        Example:
            metrics = {'ic': 0.05, 'ir': 0.8, 'sharpe': 1.5}
            logger.log_metrics(metrics)
        """
        try:
            # 添加时间戳
            metrics_with_time = {
                **metrics,
                'timestamp': datetime.now().isoformat()
            }
            path = self.storage.log(metrics_with_time, tag=tag, save_type="json")
            self.logger.info(f"✅ Logged metrics: {list(metrics.keys())} -> {path}")
            return path
        except Exception as e:
            self.logger.error(f"❌ Failed to log metrics: {e}")
            raise
    
    def log_text(self, text: str, tag: str = "limitup.log") -> Path:
        """
        记录文本日志 (text 格式)
        
        Args:
            text: 文本内容
            tag: 标签
        
        Returns:
            保存的文件路径
        """
        try:
            path = self.storage.log(text, tag=tag, save_type="text")
            self.logger.debug(f"Logged text to {path}")
            return path
        except Exception as e:
            self.logger.error(f"❌ Failed to log text: {e}")
            raise
    
    def iter_experiments(self, tag: str = "limitup.factor") -> Iterator[Any]:
        """
        迭代历史实验对象
        
        Args:
            tag: 标签 (用于过滤)
        
        Yields:
            实验对象
        
        Example:
            for exp in logger.iter_experiments():
                print(f'Hypothesis: {exp.hypothesis.hypothesis}')
                print(f'IC: {exp.result["IC"]}')
        """
        try:
            count = 0
            for msg in self.storage.iter_msg(tag=tag):
                # 只返回实验对象 (有 hypothesis 属性)
                if hasattr(msg.content, 'hypothesis'):
                    count += 1
                    yield msg.content
            
            self.logger.info(f"✅ Iterated {count} experiments from tag '{tag}'")
        except Exception as e:
            self.logger.error(f"❌ Failed to iterate experiments: {e}")
            raise
    
    def iter_metrics(self, tag: str = "limitup.metrics") -> Iterator[Dict[str, Any]]:
        """
        迭代历史指标数据
        
        Args:
            tag: 标签 (用于过滤)
        
        Yields:
            指标字典
        
        Example:
            for metrics in logger.iter_metrics():
                print(f'IC: {metrics["ic"]}, IR: {metrics["ir"]}')
        """
        try:
            count = 0
            for msg in self.storage.iter_msg(tag=tag):
                # 只返回字典类型的指标
                if isinstance(msg.content, dict):
                    count += 1
                    yield msg.content
            
            self.logger.info(f"✅ Iterated {count} metrics from tag '{tag}'")
        except Exception as e:
            self.logger.error(f"❌ Failed to iterate metrics: {e}")
            raise
    
    def get_experiment_count(self, tag: str = "limitup.factor") -> int:
        """
        获取实验数量
        
        Args:
            tag: 标签 (用于过滤)
        
        Returns:
            实验数量
        """
        try:
            count = sum(1 for _ in self.iter_experiments(tag=tag))
            return count
        except Exception:
            return 0
    
    def get_latest_experiment(self, tag: str = "limitup.factor") -> Optional[Any]:
        """
        获取最新的实验对象
        
        Args:
            tag: 标签 (用于过滤)
        
        Returns:
            最新的实验对象,如果没有则返回 None
        """
        try:
            # iter_msg 按时间戳排序,取最后一个
            latest = None
            for exp in self.iter_experiments(tag=tag):
                latest = exp
            return latest
        except Exception as e:
            self.logger.error(f"❌ Failed to get latest experiment: {e}")
            return None
    
    def clear_logs(self, tag: Optional[str] = None) -> int:
        """
        清理日志文件 (谨慎使用!)
        
        Args:
            tag: 如果指定,只清理该 tag 下的日志;否则清理所有日志
        
        Returns:
            删除的文件数量
        """
        count = 0
        try:
            if tag:
                # 清理特定 tag
                tag_path = self.workspace_path / tag.replace('.', '/')
                if tag_path.exists():
                    for file in tag_path.rglob('*.pkl'):
                        file.unlink()
                        count += 1
                    for file in tag_path.rglob('*.json'):
                        file.unlink()
                        count += 1
                    self.logger.warning(f"⚠️ Cleared {count} log files from tag '{tag}'")
            else:
                # 清理所有日志
                for file in self.workspace_path.rglob('*.pkl'):
                    file.unlink()
                    count += 1
                for file in self.workspace_path.rglob('*.json'):
                    file.unlink()
                    count += 1
                self.logger.warning(f"⚠️ Cleared ALL {count} log files")
            
            return count
        except Exception as e:
            self.logger.error(f"❌ Failed to clear logs: {e}")
            return count
    
    def __repr__(self) -> str:
        return f"QilinRDAgentLogger(workspace={self.workspace_path})"


# 便捷函数
def create_logger(workspace_path: str = './logs/rdagent') -> QilinRDAgentLogger:
    """
    创建 QilinRDAgentLogger 实例 (便捷函数)
    
    Args:
        workspace_path: 工作目录路径
    
    Returns:
        QilinRDAgentLogger 实例
    
    Example:
        logger = create_logger('./logs/my_experiment')
        logger.log_experiment(exp)
    """
    return QilinRDAgentLogger(workspace_path)


# 测试代码
if __name__ == "__main__":
    """
    测试 QilinRDAgentLogger
    
    运行方式:
        python rd_agent/logging_integration.py
    """
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("测试 QilinRDAgentLogger")
    print("=" * 60)
    
    try:
        # 创建 logger
        test_logger = create_logger('./test_logs')
        print(f"\n✅ Logger 创建成功: {test_logger}")
        
        # 测试记录指标
        print("\n测试记录指标...")
        test_metrics = {
            'ic': 0.05,
            'ir': 0.8,
            'sharpe': 1.5,
            'annual_return': 0.15
        }
        path = test_logger.log_metrics(test_metrics)
        print(f"✅ 指标已保存到: {path}")
        
        # 测试读取指标
        print("\n测试读取指标...")
        for i, metrics in enumerate(test_logger.iter_metrics()):
            print(f"  指标 {i+1}: IC={metrics.get('ic')}, IR={metrics.get('ir')}")
            if i >= 2:  # 只显示前3个
                break
        
        print("\n" + "=" * 60)
        print("✅ 测试通过!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("请确保官方 RD-Agent 已安装: pip install rdagent")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
