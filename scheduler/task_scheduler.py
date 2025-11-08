"""
定时任务调度系统
使用APScheduler实现每日自动执行交易流程
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
import logging
import sys
from pathlib import Path
import pytz

sys.path.append(str(Path(__file__).parent.parent))

try:
    from workflow.trading_workflow import TradingWorkflow
    from config.config_manager import get_config
except Exception as e:
    logging.warning(f"部分模块导入失败: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    交易定时调度器
    
    功能：
    1. T日收盘后筛选候选股票
    2. T+1日竞价时段监控竞价数据
    3. T+1日开盘后执行买入
    4. T+2日开盘后执行卖出
    5. 每日盘后数据更新和分析
    """
    
    def __init__(self, config: Optional[Dict] = None, mode: str = 'background'):
        """
        初始化调度器
        
        Parameters:
        -----------
        config: Dict
            配置字典
        mode: str
            调度器模式：'blocking'（阻塞）或 'background'（后台）
        """
        # 加载配置
        if config is None:
            config_manager = get_config()
            self.config = config_manager.to_dict()
        else:
            self.config = config
        
        # 时区设置
        self.timezone = pytz.timezone(
            self.config.get('scheduler', {}).get('timezone', 'Asia/Shanghai')
        )
        
        # 创建调度器
        if mode == 'blocking':
            self.scheduler = BlockingScheduler(timezone=self.timezone)
        else:
            self.scheduler = BackgroundScheduler(timezone=self.timezone)
        
        # 工作流实例
        self.workflow = None
        try:
            self.workflow = TradingWorkflow(config=self._prepare_workflow_config())
            logger.info("✓ 工作流实例化成功")
        except Exception as e:
            logger.warning(f"工作流实例化失败: {e}")
        
        # 任务执行历史
        self.execution_history = []
        
        # 添加事件监听
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
    
    def _prepare_workflow_config(self) -> Dict:
        """准备工作流配置"""
        return {
            'enable_t_day_screening': self.config.get('workflow', {}).get('enable_t_day_screening', True),
            'enable_t1_auction_monitor': self.config.get('workflow', {}).get('enable_t1_auction_monitor', True),
            'enable_t1_buy': self.config.get('workflow', {}).get('enable_t1_buy', True),
            'enable_t2_sell': self.config.get('workflow', {}).get('enable_t2_sell', True),
            'enable_journal': self.config.get('journal', {}).get('enable_journal', True),
            'enable_market_breaker': self.config.get('market_breaker', {}).get('enable_breaker', True),
            'enable_kelly_position': self.config.get('kelly', {}).get('enable_kelly', True),
            'screening': self.config.get('screening', {}),
            'auction': self.config.get('auction', {}),
            'buy': self.config.get('buy', {}),
            'sell': self.config.get('sell', {}),
            'risk': {
                'enable_breaker': self.config.get('market_breaker', {}).get('enable_breaker', True),
                'enable_kelly': self.config.get('kelly', {}).get('enable_kelly', True)
            }
        }
    
    def add_daily_tasks(self):
        """添加每日定时任务"""
        scheduler_config = self.config.get('scheduler', {})
        
        # Task 1: T日收盘后筛选候选（15:30）
        t_day_screening_time = scheduler_config.get('t_day_screening_time', '15:30')
        hour, minute = map(int, t_day_screening_time.split(':'))
        
        self.scheduler.add_job(
            func=self.task_t_day_screening,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=hour,
                minute=minute,
                timezone=self.timezone
            ),
            id='t_day_screening',
            name='T日候选筛选',
            replace_existing=True
        )
        logger.info(f"✓ 已添加任务: T日候选筛选 ({t_day_screening_time})")
        
        # Task 2: T+1竞价监控（09:15）
        t1_auction_time = scheduler_config.get('t1_auction_monitor_time', '09:15')
        hour, minute = map(int, t1_auction_time.split(':'))
        
        self.scheduler.add_job(
            func=self.task_t1_auction_monitor,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=hour,
                minute=minute,
                timezone=self.timezone
            ),
            id='t1_auction_monitor',
            name='T+1竞价监控',
            replace_existing=True
        )
        logger.info(f"✓ 已添加任务: T+1竞价监控 ({t1_auction_time})")
        
        # Task 3: T+1买入执行（09:30，竞价后立即买入）
        self.scheduler.add_job(
            func=self.task_t1_buy_execution,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=9,
                minute=30,
                timezone=self.timezone
            ),
            id='t1_buy_execution',
            name='T+1买入执行',
            replace_existing=True
        )
        logger.info(f"✓ 已添加任务: T+1买入执行 (09:30)")
        
        # Task 4: T+2卖出执行（09:30）
        t2_sell_time = scheduler_config.get('t2_sell_time', '09:30')
        hour, minute = map(int, t2_sell_time.split(':'))
        
        self.scheduler.add_job(
            func=self.task_t2_sell_execution,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=hour,
                minute=minute,
                timezone=self.timezone
            ),
            id='t2_sell_execution',
            name='T+2卖出执行',
            replace_existing=True
        )
        logger.info(f"✓ 已添加任务: T+2卖出执行 ({t2_sell_time})")
        
        # Task 5: 每日盘后分析（16:00）
        self.scheduler.add_job(
            func=self.task_daily_analysis,
            trigger=CronTrigger(
                day_of_week='mon-fri',
                hour=16,
                minute=0,
                timezone=self.timezone
            ),
            id='daily_analysis',
            name='每日盘后分析',
            replace_existing=True
        )
        logger.info(f"✓ 已添加任务: 每日盘后分析 (16:00)")
    
    def add_custom_task(
        self,
        func: Callable,
        trigger: str,
        task_id: str,
        task_name: str,
        **kwargs
    ):
        """
        添加自定义任务
        
        Parameters:
        -----------
        func: Callable
            任务函数
        trigger: str
            触发器类型：'cron', 'interval', 'date'
        task_id: str
            任务ID
        task_name: str
            任务名称
        **kwargs: 
            触发器参数
        """
        if trigger == 'cron':
            trigger_obj = CronTrigger(**kwargs, timezone=self.timezone)
        elif trigger == 'interval':
            from apscheduler.triggers.interval import IntervalTrigger
            trigger_obj = IntervalTrigger(**kwargs)
        elif trigger == 'date':
            from apscheduler.triggers.date import DateTrigger
            trigger_obj = DateTrigger(**kwargs)
        else:
            raise ValueError(f"不支持的触发器类型: {trigger}")
        
        self.scheduler.add_job(
            func=func,
            trigger=trigger_obj,
            id=task_id,
            name=task_name,
            replace_existing=True
        )
        logger.info(f"✓ 已添加自定义任务: {task_name}")
    
    # ==================== 任务实现 ====================
    
    def task_t_day_screening(self):
        """T日候选筛选任务"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始执行: T日候选筛选")
        logger.info(f"{'='*80}")
        
        try:
            if self.workflow:
                date = datetime.now(self.timezone).strftime('%Y-%m-%d')
                result = self.workflow.stage_t_day_screening(date)
                
                logger.info(f"✓ T日候选筛选完成")
                logger.info(f"  状态: {result['status']}")
                logger.info(f"  候选数量: {result['data'].get('candidate_count', 0)}")
                
                return result
            else:
                logger.warning("工作流未初始化")
                
        except Exception as e:
            logger.error(f"❌ T日候选筛选失败: {e}")
            raise
    
    def task_t1_auction_monitor(self):
        """T+1竞价监控任务"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始执行: T+1竞价监控")
        logger.info(f"{'='*80}")
        
        try:
            if self.workflow:
                date = datetime.now(self.timezone).strftime('%Y-%m-%d')
                result = self.workflow.stage_t1_auction_monitor(date)
                
                logger.info(f"✓ T+1竞价监控完成")
                logger.info(f"  状态: {result['status']}")
                logger.info(f"  信号数量: {result['data'].get('signal_count', 0)}")
                
                return result
            else:
                logger.warning("工作流未初始化")
                
        except Exception as e:
            logger.error(f"❌ T+1竞价监控失败: {e}")
            raise
    
    def task_t1_buy_execution(self):
        """T+1买入执行任务"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始执行: T+1买入执行")
        logger.info(f"{'='*80}")
        
        try:
            if self.workflow:
                date = datetime.now(self.timezone).strftime('%Y-%m-%d')
                result = self.workflow.stage_t1_buy_execution(date)
                
                logger.info(f"✓ T+1买入执行完成")
                logger.info(f"  状态: {result['status']}")
                logger.info(f"  订单数量: {result['data'].get('order_count', 0)}")
                
                return result
            else:
                logger.warning("工作流未初始化")
                
        except Exception as e:
            logger.error(f"❌ T+1买入执行失败: {e}")
            raise
    
    def task_t2_sell_execution(self):
        """T+2卖出执行任务"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始执行: T+2卖出执行")
        logger.info(f"{'='*80}")
        
        try:
            if self.workflow:
                date = datetime.now(self.timezone).strftime('%Y-%m-%d')
                result = self.workflow.stage_t2_sell_execution(date)
                
                logger.info(f"✓ T+2卖出执行完成")
                logger.info(f"  状态: {result['status']}")
                logger.info(f"  订单数量: {result['data'].get('order_count', 0)}")
                
                return result
            else:
                logger.warning("工作流未初始化")
                
        except Exception as e:
            logger.error(f"❌ T+2卖出执行失败: {e}")
            raise
    
    def task_daily_analysis(self):
        """每日盘后分析任务"""
        logger.info(f"\n{'='*80}")
        logger.info(f"开始执行: 每日盘后分析")
        logger.info(f"{'='*80}")
        
        try:
            if self.workflow:
                date = datetime.now(self.timezone).strftime('%Y-%m-%d')
                result = self.workflow.stage_post_trade_analysis(date)
                
                logger.info(f"✓ 每日盘后分析完成")
                logger.info(f"  状态: {result['status']}")
                
                return result
            else:
                logger.warning("工作流未初始化")
                
        except Exception as e:
            logger.error(f"❌ 每日盘后分析失败: {e}")
            raise
    
    # ==================== 调度器控制 ====================
    
    def start(self):
        """启动调度器"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("✓ 调度器已启动")
        else:
            logger.warning("调度器已在运行中")
    
    def shutdown(self, wait: bool = True):
        """
        停止调度器
        
        Parameters:
        -----------
        wait: bool
            是否等待所有任务完成
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("✓ 调度器已停止")
        else:
            logger.warning("调度器未运行")
    
    def pause_job(self, job_id: str):
        """暂停任务"""
        self.scheduler.pause_job(job_id)
        logger.info(f"✓ 任务已暂停: {job_id}")
    
    def resume_job(self, job_id: str):
        """恢复任务"""
        self.scheduler.resume_job(job_id)
        logger.info(f"✓ 任务已恢复: {job_id}")
    
    def remove_job(self, job_id: str):
        """移除任务"""
        self.scheduler.remove_job(job_id)
        logger.info(f"✓ 任务已移除: {job_id}")
    
    def get_jobs(self):
        """获取所有任务"""
        return self.scheduler.get_jobs()
    
    def print_jobs(self):
        """打印所有任务"""
        jobs = self.get_jobs()
        
        print(f"\n{'='*80}")
        print(f"定时任务列表 ({len(jobs)} 个任务)")
        print(f"{'='*80}")
        
        if not jobs:
            print("无任务")
            return
        
        for job in jobs:
            print(f"\n任务ID: {job.id}")
            print(f"  名称: {job.name}")
            print(f"  触发器: {job.trigger}")
            print(f"  下次执行: {job.next_run_time}")
        
        print(f"\n{'='*80}\n")
    
    def run_job_now(self, job_id: str):
        """立即运行任务"""
        job = self.scheduler.get_job(job_id)
        if job:
            job.func()
            logger.info(f"✓ 任务已执行: {job_id}")
        else:
            logger.error(f"任务不存在: {job_id}")
    
    def _job_executed_listener(self, event):
        """任务执行监听器"""
        job_id = event.job_id
        
        if event.exception:
            logger.error(f"任务执行失败: {job_id}")
            logger.error(f"  异常: {event.exception}")
            
            self.execution_history.append({
                'job_id': job_id,
                'time': datetime.now(self.timezone),
                'status': 'failed',
                'error': str(event.exception)
            })
        else:
            logger.info(f"任务执行成功: {job_id}")
            
            self.execution_history.append({
                'job_id': job_id,
                'time': datetime.now(self.timezone),
                'status': 'success'
            })
    
    def get_execution_history(self, limit: int = 10):
        """
        获取执行历史
        
        Parameters:
        -----------
        limit: int
            返回最近N条记录
        """
        return self.execution_history[-limit:]


# 使用示例
if __name__ == "__main__":
    print("\n" + "="*80)
    print("定时任务调度系统测试")
    print("="*80)
    
    # 创建调度器（后台模式）
    scheduler = TradingScheduler(mode='background')
    
    # 添加每日任务
    scheduler.add_daily_tasks()
    
    # 打印任务列表
    scheduler.print_jobs()
    
    # 启动调度器
    scheduler.start()
    
    print("\n调度器已启动，测试立即运行任务...")
    
    # 测试：立即运行一个任务
    print("\n测试运行: T日候选筛选")
    scheduler.run_job_now('t_day_screening')
    
    print("\n测试运行: T+1竞价监控")
    scheduler.run_job_now('t1_auction_monitor')
    
    # 查看执行历史
    print("\n执行历史:")
    for record in scheduler.get_execution_history():
        print(f"  {record['time']}: {record['job_id']} - {record['status']}")
    
    print("\n" + "="*80)
    print("✅ 调度器测试完成！")
    print("="*80)
    print("\n提示: 调度器将在后台运行，按 Ctrl+C 停止")
    
    # 保持运行
    try:
        import time
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n停止调度器...")
        scheduler.shutdown()
        print("✓ 调度器已停止")
