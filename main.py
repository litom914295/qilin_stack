#!/usr/bin/env python
"""
麒麟量化系统 - 主入口文件
一键启动系统
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import List, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from app.core.trading_context import ContextManager
from app.agents.trading_agents_impl import IntegratedDecisionAgent, MarketRegimeMarshal # v2.1 升级：引入市场风格元帅
from app.run_workflow_enhanced import TradingWorkflow, setup_logging

# 配置日志
logger = logging.getLogger("QilinMain")


class QilinTradingSystem:
    """麒麟交易系统主类"""
    
    def __init__(self, config_file: str = "config/default.yaml"):
        """
        初始化系统
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self.load_config(config_file)
        self.workflow = None
        self.is_running = False
        self.regime_marshal = MarketRegimeMarshal(self.config) # v2.1 升级：初始化元帅
        
    def load_config(self, config_file: str) -> dict:
        """加载配置文件"""
        config_path = Path(config_file)
        
        # 默认配置
        default_config = {
            "system": {
                "name": "麒麟量化系统",
                "version": "1.0.0",
                "mode": "simulation"  # simulation/paper/live
            },
            "data": {
                "sources": ["tushare", "akshare"],
                "cache_enabled": True,
                "cache_ttl": 300
            },
            "trading": {
                "max_positions": 5,
                "position_size": 0.2,
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "symbols": [
                    "000001", "000002", "000858", "002142", 
                    "002236", "300750", "300059", "600519"
                ]
            },
            "agents": {
                "parallel": True,
                "timeout": 30
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "health_check_interval": 60
            },
            "logging": {
                "level": "INFO",
                "file": "logs/qilin.log",
                "max_size": "100MB",
                "backup_count": 10
            }
        }
        
        # 如果配置文件存在，加载并合并
        if config_path.exists():
            try:
                if config_path.suffix == '.json':
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                elif config_path.suffix in ['.yaml', '.yml']:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                else:
                    file_config = {}
                    
                # 合并配置
                self._merge_config(default_config, file_config)
                
            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def _merge_config(self, base: dict, update: dict):
        """递归合并配置"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    async def initialize(self):
        """初始化系统组件"""
        logger.info("=" * 80)
        logger.info(f"{self.config['system']['name']} v{self.config['system']['version']}")
        logger.info("系统初始化中...")
        logger.info("=" * 80)
        
        # 1. 检查环境
        self._check_environment()
        
        # 2. 初始化工作流
        workflow_config = {
            'max_positions': self.config['trading']['max_positions'],
            'data_sources': self.config['data']['sources']
        }
        self.workflow = TradingWorkflow(workflow_config)
        
        # 3. 启动监控（如果启用）
        if self.config['monitoring']['enabled']:
            await self._start_monitoring()

        # 4. v2.1 升级：开盘前由元帅决定当日作战指令
        operational_command = await self.regime_marshal.get_operational_command()
        logger.info(f"市场风格元帅指令: {operational_command}")
        # 将指令注入到工作流中，工作流将使用这些动态参数
        self.workflow.set_operational_command(operational_command)
        
        logger.info("系统初始化完成")
    
    def _check_environment(self):
        """检查运行环境"""
        checks = []
        
        # 检查Python版本
        if sys.version_info < (3, 8):
            checks.append("Python版本需要3.8或更高")
        
        # 检查必要目录
        required_dirs = ['logs', 'data', 'reports', 'workspace']
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                logger.info(f"创建目录: {dir_name}")
        
        # 检查配置
        if not self.config['trading']['symbols']:
            checks.append("未配置交易股票池")
        
        if checks:
            logger.error("环境检查失败:")
            for check in checks:
                logger.error(f"  - {check}")
            sys.exit(1)
        
        logger.info("环境检查通过")
    
    async def _start_monitoring(self):
        """启动监控服务"""
        try:
            from prometheus_client import start_http_server, Counter, Gauge
            
            # 创建指标
            self.metrics = {
                'trades_total': Counter('qilin_trades_total', 'Total number of trades'),
                'active_positions': Gauge('qilin_active_positions', 'Number of active positions'),
                'system_health': Gauge('qilin_system_health', 'System health status')
            }
            
            # 启动HTTP服务器
            port = self.config['monitoring']['metrics_port']
            start_http_server(port)
            logger.info(f"Prometheus监控启动，端口: {port}")
            
            # 启动健康检查
            asyncio.create_task(self._health_check_loop())
            
        except ImportError:
            logger.warning("Prometheus客户端未安装，跳过监控")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        interval = self.config['monitoring']['health_check_interval']
        
        while self.is_running:
            try:
                # 执行健康检查
                health_status = await self._check_health()
                
                if 'system_health' in self.metrics:
                    self.metrics['system_health'].set(1 if health_status else 0)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(interval)
    
    async def _check_health(self) -> bool:
        """检查系统健康状态"""
        # 这里可以添加各种健康检查逻辑
        return True
    
    async def run(self):
        """运行主循环"""
        self.is_running = True
        mode = self.config['system']['mode']
        symbols = self.config['trading']['symbols']
        
        logger.info(f"启动交易系统 - 模式: {mode}")
        
        try:
            if mode == "live":
                await self._run_live_trading(symbols)
            elif mode == "paper":
                await self._run_paper_trading(symbols)
            else:
                await self._run_simulation(symbols)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        except Exception as e:
            logger.error(f"系统错误: {e}")
        finally:
            await self.shutdown()
    
    async def _run_live_trading(self, symbols: List[str]):
        """运行实盘交易"""
        logger.warning("实盘交易模式 - 请确保已充分测试！")
        
        while self.is_running:
            try:
                # 获取当前时间
                now = datetime.now()
                
                # 检查是否是交易时间
                if self._is_trading_time(now):
                    # 运行交易流程
                    report = await self.workflow.run(
                        symbols=symbols,
                        mode="live",
                        date=None
                    )
                    
                    # 记录交易
                    self._log_trades(report)
                
                # 等待下一个检查周期
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"实盘交易错误: {e}")
                await asyncio.sleep(60)
    
    async def _run_paper_trading(self, symbols: List[str]):
        """运行模拟盘交易"""
        logger.info("模拟盘交易模式 - 使用实时数据但不执行真实交易")
        
        while self.is_running:
            try:
                # 运行交易流程
                report = await self.workflow.run(
                    symbols=symbols,
                    mode="backtest",
                    date=None
                )
                
                # 显示结果
                self._display_results(report)
                
                # 等待
                await asyncio.sleep(300)  # 5分钟运行一次
                
            except Exception as e:
                logger.error(f"模拟盘错误: {e}")
                await asyncio.sleep(60)
    
    async def _run_simulation(self, symbols: List[str]):
        """运行模拟测试"""
        logger.info("模拟测试模式 - 使用模拟数据")
        
        # 运行一次完整的交易流程
        report = await self.workflow.run(
            symbols=symbols,
            mode="backtest",
            date=None
        )
        
        # 显示详细结果
        self._display_results(report)
        
        # 保存报告
        self._save_report(report)
    
    def _is_trading_time(self, now: datetime) -> bool:
        """检查是否是交易时间"""
        # 周一到周五
        if now.weekday() >= 5:
            return False
        
        # 交易时段: 9:30-11:30, 13:00-15:00
        time_now = now.time()
        
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()
        
        return (morning_start <= time_now <= morning_end or 
                afternoon_start <= time_now <= afternoon_end)
    
    def _log_trades(self, report: dict):
        """记录交易"""
        if 'trades' in self.metrics:
            self.metrics['trades_total'].inc(len(report.get('signals', [])))
    
    def _display_results(self, report: dict):
        """显示结果"""
        print("\n" + "=" * 80)
        print("交易报告摘要")
        print("=" * 80)
        
        summary = report.get('summary', {})
        print(f"分析股票数: {summary.get('total_analyzed', 0)}")
        print(f"生成信号数: {summary.get('total_signals', 0)}")
        print(f"执行交易数: {summary.get('total_executed', 0)}")
        
        scores = report.get('scores', {})
        print(f"平均得分: {scores.get('average', 0):.2f}")
        print(f"最高得分: {scores.get('max', 0):.2f}")
        
        # 显示Top股票
        print("\nTop 5 股票:")
        for stock in report.get('top_stocks', [])[:5]:
            print(f"  {stock['symbol']}: {stock['score']:.2f} - {stock['decision']}")
        
        print("=" * 80)
    
    def _save_report(self, report: dict):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(f"reports/report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"报告已保存: {report_file}")
    
    async def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭系统...")
        self.is_running = False
        
        # 清理资源
        # ...
        
        logger.info("系统已关闭")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="麒麟量化系统")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--mode", choices=["live", "paper", "simulation"], 
                       default="simulation", help="运行模式")
    parser.add_argument("--symbols", nargs="+", help="股票代码列表")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 创建系统实例
    system = QilinTradingSystem(args.config)
    
    # 覆盖配置
    if args.mode:
        system.config['system']['mode'] = args.mode
    if args.symbols:
        system.config['trading']['symbols'] = args.symbols
    
    # 初始化
    await system.initialize()
    
    # 运行
    await system.run()


if __name__ == "__main__":
    try:
        print("""
    ============================================
         Qilin Trading System v1.0.0
         A-Share Stock Trading System
    ============================================
        """)
    except:
        print("Qilin Trading System v1.0.0")
    
    asyncio.run(main())
