"""缠论CNN模型训练脚本

命令行工具,支持大规模训练

用法:

# 使用模拟数据训练(演示)
python scripts/train_chanlun_cnn.py --demo

# 使用真实数据训练
python scripts/train_chanlun_cnn.py \
    --stock-file data/stock_universe.txt \
    --start-date 2018-01-01 \
    --end-date 2023-12-31 \
    --epochs 100 \
    --batch-size 128 \
    --device cuda \
    --output models/chanlun_cnn.pth

# 评估已训练模型
python scripts/train_chanlun_cnn.py \
    --eval \
    --model-path models/chanlun_cnn.pth \
    --test-stocks data/test_stocks.txt
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.chanlun_dl_model import ChanLunDLTrainer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/chanlun_cnn_training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_stock_universe(file_path: str) -> list:
    """从文件加载股票列表"""
    if not os.path.exists(file_path):
        logger.warning(f"股票文件不存在: {file_path}, 使用默认列表")
        return ['000001', '600000', '000002', '000300', '600519']
    
    with open(file_path, 'r', encoding='utf-8') as f:
        stocks = [line.strip() for line in f if line.strip()]
    
    logger.info(f"加载股票列表: {len(stocks)}只")
    return stocks


def train_model(args):
    """训练模型"""
    logger.info("=" * 60)
    logger.info("缠论CNN模型训练")
    logger.info("=" * 60)
    
    # 加载股票列表
    if args.demo:
        stock_universe = ['000001', '600000', '000002']
        logger.info("演示模式: 使用3只股票")
    else:
        stock_universe = load_stock_universe(args.stock_file)
    
    # 创建训练器
    trainer = ChanLunDLTrainer(
        device=args.device,
        window_size=args.window_size
    )
    
    # 准备训练数据
    logger.info(f"准备训练数据: {args.start_date} ~ {args.end_date}")
    X, y = trainer.prepare_training_data(
        stock_universe=stock_universe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    logger.info(f"数据规模: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"标签分布: {np.bincount(y)}")
    
    # 训练模型
    logger.info(f"开始训练: epochs={args.epochs}, batch_size={args.batch_size}")
    history = trainer.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split
    )
    
    # 打印训练结果
    logger.info("-" * 60)
    logger.info("训练完成!")
    logger.info(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    logger.info(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
    
    # 保存模型
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trainer.save_model(args.output)
    logger.info(f"模型已保存: {args.output}")
    
    # 保存训练历史
    history_path = args.output.replace('.pth', '_history.npz')
    np.savez(
        history_path,
        train_loss=history['train_loss'],
        val_loss=history['val_loss'],
        val_acc=history['val_acc']
    )
    logger.info(f"训练历史已保存: {history_path}")
    
    logger.info("=" * 60)


def evaluate_model(args):
    """评估模型"""
    logger.info("=" * 60)
    logger.info("缠论CNN模型评估")
    logger.info("=" * 60)
    
    # 加载模型
    trainer = ChanLunDLTrainer(device=args.device)
    trainer.load_model(args.model_path)
    logger.info(f"模型已加载: {args.model_path}")
    
    # 加载测试数据
    test_stocks = load_stock_universe(args.test_stocks)
    logger.info(f"准备测试数据: {len(test_stocks)}只股票")
    
    X_test, y_test = trainer.prepare_training_data(
        stock_universe=test_stocks,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    logger.info(f"测试数据: X.shape={X_test.shape}, y.shape={y_test.shape}")
    
    # 预测
    predictions, probabilities = trainer.predict(X_test)
    
    # 计算准确率
    accuracy = (predictions == y_test).mean()
    logger.info(f"测试准确率: {accuracy:.4f}")
    
    # 分类报告
    from sklearn.metrics import classification_report
    label_names = ['无信号', '一买', '二买', '三买']
    report = classification_report(y_test, predictions, target_names=label_names)
    logger.info("\n分类报告:\n" + report)
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='缠论CNN模型训练脚本')
    
    # 模式选择
    parser.add_argument('--demo', action='store_true', help='演示模式(使用少量数据)')
    parser.add_argument('--eval', action='store_true', help='评估模式')
    
    # 数据参数
    parser.add_argument('--stock-file', type=str, default='data/stock_universe.txt',
                       help='股票列表文件路径')
    parser.add_argument('--test-stocks', type=str, default='data/test_stocks.txt',
                       help='测试股票列表(评估模式)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='开始日期')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='结束日期')
    
    # 模型参数
    parser.add_argument('--window-size', type=int, default=20,
                       help='时间窗口大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='训练设备')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='models/chanlun_cnn.pth',
                       help='模型输出路径')
    parser.add_argument('--model-path', type=str, default='models/chanlun_cnn.pth',
                       help='模型路径(评估模式)')
    
    args = parser.parse_args()
    
    # 创建必要目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        if args.eval:
            evaluate_model(args)
        else:
            train_model(args)
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
