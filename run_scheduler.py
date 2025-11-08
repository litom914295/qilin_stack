"""
调度器启动脚本
启动定时任务调度系统，实现自动化交易
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scheduler.task_scheduler import TradingScheduler
from config.config_manager import get_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qilin交易调度系统')
    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['blocking', 'background'],
        default='blocking',
        help='调度器模式'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式（立即运行所有任务）'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config_manager = get_config(config_file=args.config)
        
        # 验证配置
        if not config_manager.validate():
            logger.error("配置验证失败，请检查配置文件")
            sys.exit(1)
        
        # 创建调度器
        logger.info(f"创建调度器（模式: {args.mode}）")
        scheduler = TradingScheduler(
            config=config_manager.to_dict(),
            mode=args.mode
        )
        
        # 添加每日任务
        logger.info("添加每日定时任务")
        scheduler.add_daily_tasks()
        
        # 打印任务列表
        scheduler.print_jobs()
        
        # 启动调度器
        logger.info("启动调度器...")
        scheduler.start()
        
        logger.info("✓ 调度器已启动并运行")
        
        # 测试模式：立即运行所有任务
        if args.test:
            logger.info("\n=== 测试模式：立即运行所有任务 ===")
            
            jobs = scheduler.get_jobs()
            for job in jobs:
                logger.info(f"\n运行任务: {job.name}")
                scheduler.run_job_now(job.id)
            
            logger.info("\n=== 测试完成 ===")
            scheduler.shutdown()
            return
        
        # 保持运行
        logger.info("\n调度器正在运行中...")
        logger.info("按 Ctrl+C 停止调度器")
        
        if args.mode == 'blocking':
            # 阻塞模式会一直运行
            pass
        else:
            # 后台模式需要手动保持运行
            import time
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\n收到停止信号，正在关闭调度器...")
        scheduler.shutdown()
        logger.info("✓ 调度器已停止")
    
    except Exception as e:
        logger.error(f"调度器运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 确保日志目录存在
    Path("logs").mkdir(exist_ok=True)
    
    # 运行主程序
    main()
