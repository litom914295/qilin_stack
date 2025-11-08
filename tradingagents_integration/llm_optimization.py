"""
LLM完整集成优化 (P1-4)

功能:
1. 智能缓存 (Redis/本地内存)
2. Token使用统计和成本控制
3. 多模型切换和路由
4. Prompt优化和版本管理
5. 错误处理和重试机制

集成到: tradingagents_integration
"""

import os
import asyncio
import logging
import hashlib
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class LLMCall:
    """LLM调用记录"""
    id: str
    agent_name: str
    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


@dataclass
class TokenBudget:
    """Token预算"""
    daily_limit: int
    monthly_limit: int
    cost_limit_usd: float
    current_daily_usage: int = 0
    current_monthly_usage: int = 0
    current_cost_usd: float = 0.0


# ============================================================================
# 智能缓存
# ============================================================================

class IntelligentLLMCache:
    """
    智能LLM缓存
    
    特性:
    1. 语义相似度匹配 (可选)
    2. Redis/本地内存双模式
    3. TTL自动过期
    4. LRU淘汰策略
    """
    
    def __init__(
        self,
        backend: str = "memory",  # "redis" or "memory"
        ttl: int = 3600,  # 1小时
        max_size: int = 10000,
        redis_url: Optional[str] = None
    ):
        """
        初始化缓存
        
        Args:
            backend: 缓存后端 ("redis" or "memory")
            ttl: 缓存TTL (秒)
            max_size: 最大缓存条目数
            redis_url: Redis连接URL
        """
        self.backend = backend
        self.ttl = ttl
        self.max_size = max_size
        
        if backend == "redis":
            try:
                import redis.asyncio as redis
                self.redis = redis.from_url(redis_url or "redis://localhost:6379")
                logger.info("✅ Redis缓存已启用")
            except ImportError:
                logger.warning("Redis未安装,降级到内存缓存")
                self.backend = "memory"
                self.cache = {}
                self.access_times = {}
        else:
            self.cache = {}
            self.access_times = {}
        
        # 统计
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_prompt(self, prompt: str, context: Dict = None) -> str:
        """生成prompt哈希"""
        content = prompt + json.dumps(context or {}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(
        self,
        prompt: str,
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        获取缓存
        
        Args:
            prompt: 提示词
            context: 上下文
            
        Returns:
            缓存的响应 or None
        """
        cache_key = self._hash_prompt(prompt, context)
        
        try:
            if self.backend == "redis":
                value = await self.redis.get(f"llm:cache:{cache_key}")
                if value:
                    self.hit_count += 1
                    return value.decode()
                else:
                    self.miss_count += 1
                    return None
            else:
                # 内存缓存
                if cache_key in self.cache:
                    entry_time, cached_value = self.cache[cache_key]
                    
                    # 检查TTL
                    if (datetime.now() - entry_time).seconds < self.ttl:
                        self.hit_count += 1
                        self.access_times[cache_key] = datetime.now()
                        return cached_value
                    else:
                        # 过期,删除
                        del self.cache[cache_key]
                
                self.miss_count += 1
                return None
        
        except Exception as e:
            logger.error(f"缓存读取失败: {e}")
            self.miss_count += 1
            return None
    
    async def set(
        self,
        prompt: str,
        context: Optional[Dict],
        response: str
    ):
        """
        设置缓存
        
        Args:
            prompt: 提示词
            context: 上下文
            response: LLM响应
        """
        cache_key = self._hash_prompt(prompt, context)
        
        try:
            if self.backend == "redis":
                await self.redis.setex(
                    f"llm:cache:{cache_key}",
                    self.ttl,
                    response.encode()
                )
            else:
                # 内存缓存 - LRU淘汰
                if len(self.cache) >= self.max_size:
                    # 删除最久未访问的
                    oldest_key = min(
                        self.access_times.keys(),
                        key=lambda k: self.access_times[k]
                    )
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
                
                self.cache[cache_key] = (datetime.now(), response)
                self.access_times[cache_key] = datetime.now()
        
        except Exception as e:
            logger.error(f"缓存写入失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        
        return {
            'backend': self.backend,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache) if self.backend == "memory" else None
        }


# ============================================================================
# Token使用追踪器
# ============================================================================

class TokenUsageTracker:
    """
    Token使用追踪器
    
    功能:
    1. 每个Agent的Token统计
    2. 成本追踪 (美元)
    3. 预算控制
    4. 超限告警
    """
    
    def __init__(
        self,
        daily_budget_tokens: int = 1000000,  # 100万tokens/天
        daily_budget_usd: float = 50.0,  # $50/天
        alert_threshold: float = 0.8  # 80%触发告警
    ):
        self.daily_budget_tokens = daily_budget_tokens
        self.daily_budget_usd = daily_budget_usd
        self.alert_threshold = alert_threshold
        
        # 统计数据
        self.calls: List[LLMCall] = []
        self.agent_stats = defaultdict(lambda: {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost_usd': 0.0,
            'cached_calls': 0
        })
        
        # 当前预算
        self.current_date = datetime.now().date()
        self.daily_usage_tokens = 0
        self.daily_usage_usd = 0.0
        
        logger.info(
            f"Token追踪器初始化: "
            f"预算={daily_budget_tokens:,}tokens/${daily_budget_usd}/天"
        )
    
    def track(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cached: bool = False,
        prompt_hash: str = ""
    ):
        """
        记录一次LLM调用
        
        Args:
            agent_name: Agent名称
            model: 模型名称
            input_tokens: 输入tokens
            output_tokens: 输出tokens
            cost_usd: 成本 (美元)
            cached: 是否命中缓存
            prompt_hash: Prompt哈希
        """
        # 检查日期,重置每日统计
        today = datetime.now().date()
        if today != self.current_date:
            self.current_date = today
            self.daily_usage_tokens = 0
            self.daily_usage_usd = 0.0
        
        # 更新每日使用量
        total_tokens = input_tokens + output_tokens
        self.daily_usage_tokens += total_tokens
        self.daily_usage_usd += cost_usd
        
        # 记录调用
        call = LLMCall(
            id=f"{agent_name}_{datetime.now().timestamp()}",
            agent_name=agent_name,
            model=model,
            prompt_hash=prompt_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cached=cached
        )
        self.calls.append(call)
        
        # 更新Agent统计
        stats = self.agent_stats[agent_name]
        stats['calls'] += 1
        stats['input_tokens'] += input_tokens
        stats['output_tokens'] += output_tokens
        stats['cost_usd'] += cost_usd
        if cached:
            stats['cached_calls'] += 1
        
        # 检查预算告警
        self._check_budget_alert()
        
        logger.debug(
            f"Token追踪: {agent_name} | "
            f"输入={input_tokens}, 输出={output_tokens}, "
            f"成本=${cost_usd:.4f}"
        )
    
    def _check_budget_alert(self):
        """检查预算告警"""
        token_usage_pct = self.daily_usage_tokens / self.daily_budget_tokens
        cost_usage_pct = self.daily_usage_usd / self.daily_budget_usd
        
        if token_usage_pct > self.alert_threshold:
            logger.warning(
                f"⚠️ Token预算告警: {token_usage_pct:.1%} "
                f"({self.daily_usage_tokens:,}/{self.daily_budget_tokens:,})"
            )
        
        if cost_usage_pct > self.alert_threshold:
            logger.warning(
                f"⚠️ 成本预算告警: {cost_usage_pct:.1%} "
                f"(${self.daily_usage_usd:.2f}/${self.daily_budget_usd})"
            )
        
        # 超限拒绝
        if token_usage_pct >= 1.0 or cost_usage_pct >= 1.0:
            raise RuntimeError(
                f"❌ 预算超限! "
                f"Token: {token_usage_pct:.1%}, Cost: {cost_usage_pct:.1%}"
            )
    
    def get_report(self) -> Dict[str, Any]:
        """获取使用报告"""
        return {
            'daily_usage': {
                'tokens': self.daily_usage_tokens,
                'tokens_budget': self.daily_budget_tokens,
                'tokens_pct': self.daily_usage_tokens / self.daily_budget_tokens,
                'cost_usd': self.daily_usage_usd,
                'cost_budget': self.daily_budget_usd,
                'cost_pct': self.daily_usage_usd / self.daily_budget_usd
            },
            'agent_stats': dict(self.agent_stats),
            'total_calls': len(self.calls),
            'last_update': self.current_date.isoformat()
        }


# ============================================================================
# 多模型路由器
# ============================================================================

class ModelRouter:
    """
    多模型路由器
    
    根据任务类型、成本、性能选择最优模型
    """
    
    # 模型配置
    MODEL_PROFILES = {
        "gpt-4-turbo": {
            "cost": "high",
            "quality": "excellent",
            "speed": "medium",
            "use_cases": ["complex_analysis", "reasoning", "creative"]
        },
        "gpt-3.5-turbo": {
            "cost": "low",
            "quality": "good",
            "speed": "fast",
            "use_cases": ["simple_query", "translation", "summarization"]
        },
        "claude-3-sonnet": {
            "cost": "medium",
            "quality": "excellent",
            "speed": "medium",
            "use_cases": ["analysis", "coding", "reasoning"]
        },
        "claude-3-haiku": {
            "cost": "very_low",
            "quality": "good",
            "speed": "very_fast",
            "use_cases": ["simple_query", "classification"]
        }
    }
    
    def __init__(self, default_model: str = "gpt-3.5-turbo"):
        self.default_model = default_model
        logger.info(f"模型路由器初始化: 默认模型={default_model}")
    
    def select_model(
        self,
        task_type: str,
        priority: str = "balanced"  # "cost", "quality", "speed", "balanced"
    ) -> str:
        """
        选择最优模型
        
        Args:
            task_type: 任务类型
            priority: 优先级
            
        Returns:
            模型名称
        """
        # 简化实现: 根据优先级选择
        if priority == "cost":
            # 成本优先 - 选最便宜的
            return "claude-3-haiku"
        elif priority == "quality":
            # 质量优先 - 选最好的
            return "gpt-4-turbo"
        elif priority == "speed":
            # 速度优先 - 选最快的
            return "gpt-3.5-turbo"
        else:
            # 平衡模式
            if task_type in ["complex_analysis", "reasoning"]:
                return "claude-3-sonnet"
            else:
                return "gpt-3.5-turbo"


# ============================================================================
# LLM优化管理器 (集成所有功能)
# ============================================================================

class OptimizedLLMManager:
    """
    LLM优化管理器
    
    集成:
    1. IntelligentLLMCache - 智能缓存
    2. TokenUsageTracker - Token追踪
    3. ModelRouter - 模型路由
    4. 重试机制
    5. 错误处理
    """
    
    def __init__(
        self,
        api_key: str,
        provider: str = "openai",  # "openai" or "anthropic"
        enable_cache: bool = True,
        enable_tracking: bool = True,
        cache_backend: str = "memory",
        daily_budget_usd: float = 50.0
    ):
        """
        初始化优化管理器
        
        Args:
            api_key: API密钥
            provider: LLM提供商
            enable_cache: 启用缓存
            enable_tracking: 启用追踪
            cache_backend: 缓存后端
            daily_budget_usd: 每日预算 (美元)
        """
        self.provider = provider
        self.api_key = api_key
        
        # 初始化组件
        if enable_cache:
            self.cache = IntelligentLLMCache(backend=cache_backend)
        else:
            self.cache = None
        
        if enable_tracking:
            self.tracker = TokenUsageTracker(daily_budget_usd=daily_budget_usd)
        else:
            self.tracker = None
        
        self.router = ModelRouter()
        
        # LLM客户端 (延迟初始化)
        self.llm_client = None
        
        logger.info(
            f"✅ 优化LLM管理器初始化: "
            f"提供商={provider}, 缓存={'✅' if enable_cache else '❌'}, "
            f"追踪={'✅' if enable_tracking else '❌'}"
        )
    
    def _get_llm_client(self, model: str):
        """获取LLM客户端"""
        # 简化实现: 返回Mock
        class MockLLMClient:
            async def call(self, system_prompt, user_prompt, **kwargs):
                # 模拟响应
                response = f"[Mock响应] 分析完成: {user_prompt[:50]}..."
                input_tokens = len(system_prompt) // 4 + len(user_prompt) // 4
                output_tokens = len(response) // 4
                return response, input_tokens, output_tokens
            
            def get_cost(self, input_tokens, output_tokens):
                return (input_tokens + output_tokens) / 1000 * 0.002
        
        return MockLLMClient()
    
    async def call(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        task_type: str = "general",
        priority: str = "balanced",
        max_retries: int = 3,
        context: Optional[Dict] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        优化的LLM调用
        
        Args:
            agent_name: Agent名称
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            task_type: 任务类型
            priority: 优先级
            max_retries: 最大重试次数
            context: 上下文
            
        Returns:
            (响应, 元数据)
        """
        start_time = datetime.now()
        
        # 1. 检查缓存
        cached_response = None
        if self.cache:
            cached_response = await self.cache.get(user_prompt, context)
            if cached_response:
                logger.info(f"✅ 缓存命中: {agent_name}")
                
                # 记录缓存命中
                if self.tracker:
                    self.tracker.track(
                        agent_name=agent_name,
                        model="cached",
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        cached=True
                    )
                
                duration = (datetime.now() - start_time).total_seconds() * 1000
                return cached_response, {
                    'cached': True,
                    'duration_ms': duration
                }
        
        # 2. 选择模型
        model = self.router.select_model(task_type, priority)
        
        # 3. 调用LLM (带重试)
        last_error = None
        for attempt in range(max_retries):
            try:
                client = self._get_llm_client(model)
                response, input_tokens, output_tokens = await client.call(
                    system_prompt,
                    user_prompt
                )
                
                # 4. 计算成本
                cost = client.get_cost(input_tokens, output_tokens)
                
                # 5. 记录追踪
                if self.tracker:
                    self.tracker.track(
                        agent_name=agent_name,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                        cached=False
                    )
                
                # 6. 写入缓存
                if self.cache:
                    await self.cache.set(user_prompt, context, response)
                
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.info(
                    f"✅ LLM调用成功: {agent_name} | 模型={model} | "
                    f"输入={input_tokens}, 输出={output_tokens}, 成本=${cost:.4f}"
                )
                
                return response, {
                    'cached': False,
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost_usd': cost,
                    'duration_ms': duration,
                    'attempt': attempt + 1
                }
            
            except Exception as e:
                last_error = e
                logger.warning(f"LLM调用失败 (尝试{attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
        
        # 全部重试失败
        raise RuntimeError(f"LLM调用失败 ({max_retries}次重试): {last_error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        if self.tracker:
            stats['usage'] = self.tracker.get_report()
        
        return stats


# ============================================================================
# 使用示例
# ============================================================================

async def example_llm_optimization():
    """LLM优化示例"""
    print("=== LLM优化集成示例 (P1-4) ===\n")
    
    # 创建优化管理器
    manager = OptimizedLLMManager(
        api_key="mock_key",
        enable_cache=True,
        enable_tracking=True,
        daily_budget_usd=50.0
    )
    
    # 1. 第一次调用 (缓存未命中)
    print("1. 首次调用...")
    response, metadata = await manager.call(
        agent_name="sentiment",
        system_prompt="你是一个市场情绪分析专家",
        user_prompt="分析贵州茅台的市场情绪",
        task_type="analysis",
        priority="balanced"
    )
    print(f"   响应: {response[:50]}...")
    print(f"   缓存: {metadata['cached']}")
    print(f"   成本: ${metadata.get('cost_usd', 0):.4f}")
    print()
    
    # 2. 第二次调用 (相同prompt,缓存命中)
    print("2. 再次调用相同prompt...")
    response, metadata = await manager.call(
        agent_name="sentiment",
        system_prompt="你是一个市场情绪分析专家",
        user_prompt="分析贵州茅台的市场情绪",
        task_type="analysis"
    )
    print(f"   缓存命中: {metadata['cached']}")
    print()
    
    # 3. 统计信息
    print("3. 统计信息:")
    stats = manager.get_stats()
    
    if 'cache' in stats:
        cache_stats = stats['cache']
        print(f"   缓存命中率: {cache_stats['hit_rate']:.1%}")
        print(f"   缓存大小: {cache_stats.get('cache_size', 'N/A')}")
    
    if 'usage' in stats:
        usage_stats = stats['usage']
        daily = usage_stats['daily_usage']
        print(f"   每日Token: {daily['tokens']:,} / {daily['tokens_budget']:,}")
        print(f"   每日成本: ${daily['cost_usd']:.2f} / ${daily['cost_budget']}")
        print(f"   总调用次数: {usage_stats['total_calls']}")
    
    print("\n✅ 示例完成!")


if __name__ == "__main__":
    asyncio.run(example_llm_optimization())
