"""
API 文档生成脚本
自动从代码注释生成 Markdown 格式的 API 文档
"""

import ast
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys


class DocGenerator:
    """文档生成器"""
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        初始化文档生成器
        
        Args:
            source_dir: 源代码目录
            output_dir: 输出文档目录
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_docs(self):
        """生成所有文档"""
        print(f"正在扫描源代码目录: {self.source_dir}")
        
        # 查找所有 Python 文件
        python_files = list(self.source_dir.rglob("*.py"))
        print(f"找到 {len(python_files)} 个 Python 文件")
        
        # 生成文档索引
        index_content = ["# Qilin Stack API 文档\n", "## 模块列表\n"]
        
        for py_file in sorted(python_files):
            # 跳过测试和临时文件
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                doc = self._generate_module_doc(py_file)
                if doc:
                    # 保存模块文档
                    relative_path = py_file.relative_to(self.source_dir)
                    doc_path = self.output_dir / relative_path.with_suffix('.md')
                    doc_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(doc)
                    
                    # 添加到索引
                    module_name = str(relative_path.with_suffix('')).replace('\\', '.')
                    index_content.append(f"- [{module_name}]({relative_path.with_suffix('.md')})\n")
                    
                    print(f"✓ 生成文档: {relative_path}")
            except Exception as e:
                print(f"✗ 处理文件失败 {py_file}: {e}")
        
        # 保存索引
        with open(self.output_dir / 'index.md', 'w', encoding='utf-8') as f:
            f.writelines(index_content)
        
        print(f"\n文档生成完成! 输出目录: {self.output_dir}")
    
    def _generate_module_doc(self, file_path: Path) -> Optional[str]:
        """
        生成单个模块的文档
        
        Args:
            file_path: 模块文件路径
            
        Returns:
            Markdown 格式的文档字符串
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        
        # 模块名称
        module_name = file_path.stem
        doc_lines = [f"# {module_name}\n\n"]
        
        # 模块文档字符串
        module_doc = ast.get_docstring(tree)
        if module_doc:
            doc_lines.append(f"{module_doc}\n\n")
        
        # 提取类和函数
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self._extract_class_info(node))
            elif isinstance(node, ast.FunctionDef):
                # 只提取顶层函数
                if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    functions.append(self._extract_function_info(node))
        
        # 生成类文档
        if classes:
            doc_lines.append("## 类\n\n")
            for cls_info in classes:
                doc_lines.extend(self._format_class_doc(cls_info))
        
        # 生成函数文档
        if functions:
            doc_lines.append("## 函数\n\n")
            for func_info in functions:
                doc_lines.extend(self._format_function_doc(func_info))
        
        return ''.join(doc_lines) if len(doc_lines) > 2 else None
    
    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """提取类信息"""
        info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'bases': [self._get_name(base) for base in node.bases],
            'methods': []
        }
        
        # 提取方法
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                info['methods'].append(self._extract_function_info(item))
        
        return info
    
    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """提取函数信息"""
        info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'args': [],
            'returns': None
        }
        
        # 提取参数
        for arg in node.args.args:
            arg_info = {'name': arg.arg}
            
            # 提取类型注解
            if arg.annotation:
                arg_info['type'] = self._get_annotation(arg.annotation)
            
            info['args'].append(arg_info)
        
        # 提取返回类型
        if node.returns:
            info['returns'] = self._get_annotation(node.returns)
        
        return info
    
    def _format_class_doc(self, cls_info: Dict[str, Any]) -> List[str]:
        """格式化类文档"""
        lines = []
        
        # 类名和继承
        bases = ', '.join(cls_info['bases']) if cls_info['bases'] else ''
        if bases:
            lines.append(f"### {cls_info['name']}({bases})\n\n")
        else:
            lines.append(f"### {cls_info['name']}\n\n")
        
        # 类文档字符串
        if cls_info['docstring']:
            lines.append(f"{cls_info['docstring']}\n\n")
        
        # 方法
        if cls_info['methods']:
            lines.append("#### 方法\n\n")
            for method in cls_info['methods']:
                # 跳过私有方法
                if method['name'].startswith('_') and not method['name'].startswith('__'):
                    continue
                
                lines.extend(self._format_function_doc(method, indent=True))
        
        return lines
    
    def _format_function_doc(self, func_info: Dict[str, Any], indent: bool = False) -> List[str]:
        """格式化函数文档"""
        lines = []
        prefix = "##### " if indent else "### "
        
        # 函数签名
        args_str = ', '.join(
            f"{arg['name']}: {arg.get('type', 'Any')}" 
            for arg in func_info['args']
        )
        returns_str = f" -> {func_info['returns']}" if func_info['returns'] else ""
        
        lines.append(f"{prefix}{func_info['name']}({args_str}){returns_str}\n\n")
        
        # 函数文档字符串
        if func_info['docstring']:
            lines.append(f"```\n{func_info['docstring']}\n```\n\n")
        
        return lines
    
    def _get_name(self, node: ast.AST) -> str:
        """获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _get_annotation(self, node: ast.AST) -> str:
        """获取类型注解"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            base = self._get_annotation(node.value)
            slice_val = self._get_annotation(node.slice)
            return f"{base}[{slice_val}]"
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation(node.value)}.{node.attr}"
        elif isinstance(node, ast.Tuple):
            elements = ', '.join(self._get_annotation(e) for e in node.elts)
            return f"({elements})"
        else:
            return "Any"


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "app"
    output_dir = project_root / "docs" / "api"
    
    # 生成文档
    generator = DocGenerator(str(source_dir), str(output_dir))
    generator.generate_docs()
    
    print("\n✓ 文档生成完成!")
    print(f"  查看文档索引: {output_dir / 'index.md'}")


if __name__ == '__main__':
    main()
