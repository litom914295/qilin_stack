#!/usr/bin/env python
"""
LLM驱动的涨停板因子自动发现系统
使用 DeepSeek 自动生成和评估新因子
Windows 完全兼容，无需 Docker
"""

import asyncio
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# LLM 客户端
from openai import AsyncOpenAI

# 代码沙盒 (P1-3)
from rd_agent.code_sandbox import execute_safe, SecurityLevel

logger = logging.getLogger(__name__)


class LLMFactorDiscovery:
    """
    LLM驱动的因子自动发现系统
    
    功能：
    1. 根据市场特征自动生成新因子
    2. 生成因子的可执行代码
    3. 评估因子质量
    4. 持续迭代优化
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "deepseek-chat",
        cache_dir: str = "./workspace/llm_factor_cache"
    ):
        """
        初始化 LLM 因子发现系统
        
        Args:
            api_key: API密钥，默认从环境变量读取
            api_base: API基础URL
            model: 使用的模型
            cache_dir: 缓存目录
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
        self.model = model
        
        # 创建 OpenAI 客户端（兼容 DeepSeek）
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        # 缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 因子生成历史
        self.generation_history: List[Dict] = []
        
        logger.info(f"✅ LLM因子发现系统初始化成功")
        logger.info(f"   模型: {self.model}")
        logger.info(f"   API: {self.api_base}")
    
    async def discover_new_factors(
        self,
        n_factors: int = 5,
        focus_areas: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        自动发现新因子
        
        Args:
            n_factors: 要生成的因子数量
            focus_areas: 关注领域 ['封板', '连板', '题材', '资金', '时机']
            context: 额外的上下文信息
            
        Returns:
            新因子列表
        """
        logger.info(f"🤖 开始LLM驱动因子发现，目标生成 {n_factors} 个因子")
        
        # 构建提示词
        prompt = self._build_discovery_prompt(n_factors, focus_areas, context)
        
        # 调用 LLM
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,  # 提高创造性
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            
            # 解析因子
            factors = self._parse_factors_from_response(content)
            
            # 验证和清理
            valid_factors = []
            for factor in factors:
                if self._validate_factor(factor):
                    valid_factors.append(factor)
            
            logger.info(f"✅ 成功生成 {len(valid_factors)} 个有效因子")
            
            # 保存到历史
            self._save_generation_history(prompt, content, valid_factors)
            
            return valid_factors
            
        except Exception as e:
            logger.error(f"❌ LLM调用失败: {e}")
            return []
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的量化因子研究专家，专注于A股涨停板"一进二"策略。

你的任务是设计新的量化因子来预测：
- 今日涨停的股票，明日是否继续涨停（一进二）
- 明日的收益率和涨幅

关键考虑因素：
1. 封板质量：封单强度、封板时间、开板次数
2. 连板高度：首板、二板、三板等不同高度的特征
3. 题材热度：所属概念、板块联动、龙头地位
4. 资金行为：大单流向、换手率、分时形态
5. 时机选择：涨停时间、竞价表现、尾盘强度

要求：
- 因子必须可计算、可实现
- 提供明确的数学表达式
- 给出Python代码实现
- 说明因子的投资逻辑
- 估计预期的IC值（信息系数）

输出格式（JSON）：
```json
{
  "factors": [
    {
      "name": "因子名称",
      "expression": "数学表达式",
      "code": "Python代码",
      "category": "类别",
      "logic": "投资逻辑说明",
      "expected_ic": 0.XX,
      "data_requirements": ["字段1", "字段2"]
    }
  ]
}
```
"""
    
    def _build_discovery_prompt(
        self,
        n_factors: int,
        focus_areas: Optional[List[str]],
        context: Optional[str]
    ) -> str:
        """构建发现提示词"""
        prompt = f"请为A股涨停板'一进二'策略设计 {n_factors} 个新的量化因子。\n\n"
        
        if focus_areas:
            areas_text = "、".join(focus_areas)
            prompt += f"重点关注以下领域：{areas_text}\n\n"
        
        if context:
            prompt += f"额外上下文：{context}\n\n"
        
        prompt += """
要求：
1. 因子要有创新性，不是简单的价量指标
2. 充分考虑涨停板的特殊性（价格封死、成交受限）
3. 结合A股市场特点（T+1、涨跌停限制、情绪驱动）
4. 提供完整的实现代码
5. 估计合理的IC值（通常0.05-0.15）

请以JSON格式输出因子列表。
"""
        return prompt
    
    def _parse_factors_from_response(self, content: str) -> List[Dict[str, Any]]:
        """从LLM响应中解析因子"""
        factors = []
        
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                factors = data.get('factors', [])
            else:
                # 尝试直接解析
                json_match = re.search(r'\{.*"factors".*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    factors = data.get('factors', [])
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            # 尝试逐个提取因子
            factors = self._parse_factors_fallback(content)
        
        return factors
    
    def _parse_factors_fallback(self, content: str) -> List[Dict[str, Any]]:
        """备用解析方法"""
        factors = []
        
        # 按段落分割
        sections = content.split('\n\n')
        
        current_factor = {}
        for section in sections:
            section = section.strip()
            
            # 识别因子名称
            if '因子名称' in section or 'name' in section.lower():
                if current_factor:
                    factors.append(current_factor)
                    current_factor = {}
                
                name_match = re.search(r'[:：](.*?)(?:\n|$)', section)
                if name_match:
                    current_factor['name'] = name_match.group(1).strip()
            
            # 识别其他字段
            if 'expression' in section.lower() or '表达式' in section:
                expr_match = re.search(r'[:：](.*?)(?:\n|$)', section)
                if expr_match:
                    current_factor['expression'] = expr_match.group(1).strip()
            
            if 'code' in section.lower() or '代码' in section:
                code_match = re.search(r'```python\s*(.*?)\s*```', section, re.DOTALL)
                if code_match:
                    current_factor['code'] = code_match.group(1).strip()
        
        if current_factor:
            factors.append(current_factor)
        
        return factors
    
    def _validate_factor(self, factor: Dict[str, Any]) -> bool:
        """验证因子有效性 (P1-3: 增强安全检查)"""
        required_fields = ['name', 'expression', 'code']
        
        for field in required_fields:
            if field not in factor or not factor[field]:
                logger.warning(f"因子缺少必需字段: {field}")
                return False
        
        # P1-3: 使用代码沙监进行安全验证
        # 这里只是验证，不执行，所以只做语法检查
        code = factor.get('code', '')
        
        try:
            # 语法检查
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning(f"因子代码语法错误: {e}")
            return False
        
        # 基础关键字检查（作为额外的快速检查）
        dangerous_keywords = ['import os', 'import sys', 'import subprocess', 
                             'exec(', 'eval(', '__import__', 'open(']
        
        for keyword in dangerous_keywords:
            if keyword in code:
                logger.warning(f"因子代码包含危险关键字: {keyword}")
                return False
        
        return True
    
    async def evaluate_factor(
        self,
        factor: Dict[str, Any],
        sample_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        评估因子质量
        
        Args:
            factor: 因子定义
            sample_data: 样本数据用于测试
            
        Returns:
            评估结果
        """
        logger.info(f"📊 评估因子: {factor['name']}")
        
        evaluation = {
            'factor_name': factor['name'],
            'syntax_valid': False,
            'computable': False,
            'estimated_ic': factor.get('expected_ic', 0),
            'issues': []
        }
        
        # 1. 语法检查
        try:
            compile(factor['code'], '<string>', 'exec')
            evaluation['syntax_valid'] = True
        except SyntaxError as e:
            evaluation['issues'].append(f"语法错误: {e}")
            logger.warning(f"因子语法错误: {e}")
        
        # 2. 可计算性测试 (P1-3: 使用代码沙盒)
        if sample_data is not None and evaluation['syntax_valid']:
            try:
                # P1-3: 使用代码沙盒执行
                context = {
                    'np': np,
                    'pd': pd
                }
                
                # 添加数据列到上下文
                for col in sample_data.columns:
                    context[col] = sample_data[col]
                
                # 安全执行代码
                execution_result = execute_safe(
                    code=factor['code'],
                    context=context,
                    timeout=10
                )
                
                if execution_result.success:
                    evaluation['computable'] = True
                else:
                    evaluation['issues'].append(f"计算错误: {execution_result.error}")
                    logger.warning(f"因子计算错误: {execution_result.error}")
                
            except Exception as e:
                evaluation['issues'].append(f"计算错误: {e}")
                logger.warning(f"因子计算错误: {e}")
        
        # 3. LLM 质量评估
        if evaluation['syntax_valid']:
            quality_score = await self._llm_quality_assessment(factor)
            evaluation['quality_score'] = quality_score
        
        return evaluation
    
    async def _llm_quality_assessment(self, factor: Dict[str, Any]) -> float:
        """使用LLM评估因子质量"""
        prompt = f"""
请评估以下涨停板因子的质量（0-10分）：

因子名称: {factor['name']}
表达式: {factor['expression']}
投资逻辑: {factor.get('logic', 'N/A')}

评估标准：
1. 投资逻辑是否合理（3分）
2. 实现是否清晰（2分）
3. 创新性（2分）
4. 可计算性（2分）
5. 实用价值（1分）

请只返回一个0-10的数字分数。
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            score = float(re.search(r'\d+\.?\d*', content).group())
            return min(max(score, 0), 10)  # 限制在0-10
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            return 5.0  # 默认中等分数
    
    async def refine_factor(
        self,
        factor: Dict[str, Any],
        feedback: str
    ) -> Dict[str, Any]:
        """
        根据反馈改进因子
        
        Args:
            factor: 原始因子
            feedback: 改进建议
            
        Returns:
            改进后的因子
        """
        logger.info(f"🔄 改进因子: {factor['name']}")
        
        prompt = f"""
请改进以下涨停板因子：

原因子：
- 名称: {factor['name']}
- 表达式: {factor['expression']}
- 代码: {factor['code']}

反馈意见：
{feedback}

请提供改进后的因子，以JSON格式输出。
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            refined_factors = self._parse_factors_from_response(content)
            
            if refined_factors:
                return refined_factors[0]
            else:
                return factor
                
        except Exception as e:
            logger.error(f"因子改进失败: {e}")
            return factor
    
    def _save_generation_history(
        self,
        prompt: str,
        response: str,
        factors: List[Dict[str, Any]]
    ):
        """保存生成历史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'factors_generated': len(factors),
            'factors': factors
        }
        
        self.generation_history.append(history_entry)
        
        # 保存到文件
        history_file = self.cache_dir / f"generation_history_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_entry, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 生成历史已保存: {history_file}")
    
    def export_factors(
        self,
        factors: List[Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> str:
        """
        导出因子
        
        Args:
            factors: 因子列表
            output_file: 输出文件路径
            
        Returns:
            导出文件路径
        """
        if output_file is None:
            output_file = self.cache_dir / f"factors_export_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'total_factors': len(factors),
            'factors': factors
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📤 因子已导出: {output_file}")
        return str(output_file)


# 演示使用
async def demo():
    """演示LLM驱动的因子发现"""
    print("=" * 70)
    print("LLM驱动涨停板因子自动发现演示")
    print("=" * 70)
    
    # 创建发现系统
    discovery = LLMFactorDiscovery()
    
    # 1. 自动发现因子
    print("\n🤖 步骤1: 自动发现新因子...")
    factors = await discovery.discover_new_factors(
        n_factors=3,
        focus_areas=["封板强度", "连板动量", "题材共振"],
        context="重点关注短线强势特征"
    )
    
    print(f"\n✅ 发现 {len(factors)} 个新因子:")
    for i, factor in enumerate(factors, 1):
        print(f"\n--- 因子 {i} ---")
        print(f"名称: {factor['name']}")
        print(f"表达式: {factor['expression']}")
        print(f"逻辑: {factor.get('logic', 'N/A')[:100]}...")
        if 'expected_ic' in factor:
            print(f"预期IC: {factor['expected_ic']:.4f}")
    
    # 2. 评估因子
    if factors:
        print(f"\n📊 步骤2: 评估因子质量...")
        for factor in factors[:2]:  # 评估前2个
            evaluation = await discovery.evaluate_factor(factor)
            print(f"\n因子: {factor['name']}")
            print(f"  语法正确: {evaluation['syntax_valid']}")
            print(f"  可计算: {evaluation['computable']}")
            if 'quality_score' in evaluation:
                print(f"  质量分数: {evaluation['quality_score']:.1f}/10")
            if evaluation['issues']:
                print(f"  问题: {', '.join(evaluation['issues'])}")
    
    # 3. 导出因子
    if factors:
        print(f"\n💾 步骤3: 导出因子...")
        export_path = discovery.export_factors(factors)
        print(f"已导出到: {export_path}")


if __name__ == '__main__':
    asyncio.run(demo())
