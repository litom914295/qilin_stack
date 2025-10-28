"""
异常处理规范化脚本
自动修复裸露的except语句
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExceptionHandlingFixer:
    """异常处理修复器"""
    
    def __init__(self, target_dirs: List[str]):
        self.target_dirs = target_dirs
        self.fixes_count = 0
        self.files_processed = 0
        
    def process_files(self):
        """处理所有Python文件"""
        for target_dir in self.target_dirs:
            for py_file in Path(target_dir).rglob("*.py"):
                if self._should_skip_file(py_file):
                    continue
                    
                try:
                    self._process_file(py_file)
                    self.files_processed += 1
                except Exception as e:
                    logger.error(f"处理文件失败 {py_file}: {e}")
        
        logger.info(f"\n✅ 完成！处理了 {self.files_processed} 个文件，修复了 {self.fixes_count} 处异常处理")
    
    def _should_skip_file(self, filepath: Path) -> bool:
        """是否应跳过文件"""
        skip_patterns = [
            '.qilin',
            '__pycache__',
            '.git',
            'venv',
            'node_modules',
            'build',
            'dist',
            '.pytest_cache'
        ]
        
        return any(pattern in str(filepath) for pattern in skip_patterns)
    
    def _process_file(self, filepath: Path):
        """处理单个文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复模式1: except: pass
        content, count1 = self._fix_bare_except_pass(content)
        
        # 修复模式2: except Exception: pass  
        content, count2 = self._fix_exception_pass(content)
        
        # 修复模式3: except: 后面没有具体处理
        content, count3 = self._fix_bare_except(content)
        
        total_fixes = count1 + count2 + count3
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.fixes_count += total_fixes
            logger.info(f"✅ {filepath.name}: 修复 {total_fixes} 处")
    
    def _fix_bare_except_pass(self, content: str) -> Tuple[str, int]:
        """修复 except: pass 模式"""
        pattern = r'(\s+)except:\s*\n\s+pass'
        
        def replacer(match):
            indent = match.group(1)
            return f'''{indent}except Exception as e:
{indent}    logger.warning(f"操作失败: {{e}}")'''
        
        new_content, count = re.subn(pattern, replacer, content)
        return new_content, count
    
    def _fix_exception_pass(self, content: str) -> Tuple[str, int]:
        """修复 except Exception: pass 模式"""
        pattern = r'(\s+)except Exception:\s*\n\s+pass'
        
        def replacer(match):
            indent = match.group(1)
            return f'''{indent}except Exception as e:
{indent}    logger.warning(f"操作失败: {{e}}")'''
        
        new_content, count = re.subn(pattern, replacer, content)
        return new_content, count
    
    def _fix_bare_except(self, content: str) -> Tuple[str, int]:
        """修复裸露的 except: 模式（后面有内容的）"""
        # 这个比较复杂，只处理简单情况
        pattern = r'(\s+)except:\s*\n(\s+)([^#\n])'
        
        def replacer(match):
            indent1 = match.group(1)
            indent2 = match.group(2)
            first_char = match.group(3)
            return f'{indent1}except Exception as e:\n{indent2}{first_char}'
        
        new_content, count = re.subn(pattern, replacer, content)
        return new_content, count


def fix_specific_files():
    """修复特定已知有问题的文件"""
    fixes = {
        'app/core/cache_manager.py': [
            (89, 90, '''                except (IOError, pickle.PickleError) as e:
                    logger.debug(f"读取磁盘缓存失败: {e}")'''),
            (131, 132, '''            except (IOError, pickle.PickleError) as e:
                logger.debug(f"保存磁盘缓存失败: {e}")'''),
            (182, 183, '''            except (IOError, pickle.PickleError) as e:
                logger.debug(f"清理缓存文件失败: {e}")'''),
        ]
    }
    
    for filepath, line_fixes in fixes.items():
        full_path = Path('G:/test/qilin_stack') / filepath
        if not full_path.exists():
            logger.warning(f"文件不存在: {filepath}")
            continue
        
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 按行号倒序处理，避免行号偏移
        for start_line, end_line, new_code in sorted(line_fixes, reverse=True):
            # 删除旧的except块
            del lines[start_line-1:end_line-1]
            # 插入新代码
            lines.insert(start_line-1, new_code + '\n')
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info(f"✅ 手动修复: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("异常处理规范化工具")
    print("=" * 60)
    
    # 首先修复特定文件
    print("\n📝 修复已知问题文件...")
    fix_specific_files()
    
    # 然后批量处理
    print("\n🔍 批量扫描修复...")
    target_directories = [
        'G:/test/qilin_stack/app',
        'G:/test/qilin_stack/layer2_qlib',
        'G:/test/qilin_stack/qilin_stack',
        'G:/test/qilin_stack/tradingagents_integration',
    ]
    
    fixer = ExceptionHandlingFixer(target_directories)
    fixer.process_files()
    
    print("\n" + "=" * 60)
    print("✅ 异常处理规范化完成！")
    print("=" * 60)
    print("\n建议:")
    print("1. 运行测试确保没有破坏功能")
    print("2. Review修改的文件")
    print("3. 根据实际情况调整异常类型")
