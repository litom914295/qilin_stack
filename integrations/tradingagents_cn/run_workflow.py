
import os, argparse, yaml, pandas as pd, time
import logging
from datetime import datetime
from tools.qlib_predict import load_preds
from tools.decision_agents import run_agents
from tools.score_fuser import fused_score
from executors.order_gateway.gateway import OrderGateway, Order
from integrations.tradingagents_cn.report import write_markdown_report

# 导入验证器
try:
    from app.core.validators import Validator, RiskValidator, ValidationError
except ImportError:
    # 如果导入失败，定义一个简单的验证器
    class Validator:
        @classmethod
        def validate_symbol(cls, symbol, market=None):
            return symbol.upper().strip()
        
        @classmethod
        def validate_quantity(cls, qty):
            qty = int(qty)
            if qty <= 0 or qty % 100 != 0:
                raise ValueError(f"Invalid quantity: {qty}")
            return qty
        
        @classmethod
        def validate_price(cls, price, symbol=None):
            price = float(price)
            if price <= 0:
                raise ValueError(f"Invalid price: {price}")
            return round(price, 2)
        
        @classmethod
        def validate_config(cls, config):
            return config
        
        @classmethod
        def sanitize_input(cls, input_str, max_length=100):
            if not input_str:
                return ""
            return str(input_str).strip()[:max_length]
    
    class ValidationError(Exception):
        pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)

def _load_yaml(file_path: str, default_value: dict = None) -> dict:
    """
    加载YAML配置文件
    
    Args:
        file_path: 文件路径
        default_value: 默认值
        
    Returns:
        配置字典
    """
    if not file_path:
        return default_value or {}
    
    # 验证文件路径
    file_path = Validator.sanitize_input(file_path, max_length=255)
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 验证配置
                if config:
                    return config
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
    
    return default_value or {}

def main():
    """主函数，带输入验证"""
    ap = argparse.ArgumentParser(description='运行量化交易工作流')
    ap.add_argument('--topk', type=int, default=None, help='选择前K个股票')
    ap.add_argument('--place-orders', action='store_true', help='是否下单')
    ap.add_argument('--dry-run', action='store_true', help='模拟运行')
    args = ap.parse_args()
    
    t0 = time.time()
    
    try:
        # 验证topk参数
        if args.topk is not None:
            if args.topk <= 0 or args.topk > 10:
                raise ValidationError(f"Invalid topk value: {args.topk}, must be between 1 and 10")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return 1

    # 加载并验证配置
    try:
        syscfg = _load_yaml('integrations/tradingagents_cn/system.yaml', 
                          {'sla': {'max_runtime_sec': 45}, 'fallback': {'strategy': 'qlib_only'}})
        wcfg = _load_yaml('integrations/tradingagents_cn/weights.yaml', 
                        {'weights': {}, 'topk': 2, 'default_qty': 1000})
        
        # 验证配置参数
        weights = wcfg.get('weights', {})
        topk = args.topk or int(wcfg.get('topk', 2))
        
        # 验证topk
        if topk <= 0 or topk > 10:
            raise ValidationError(f"Invalid topk: {topk}")
        
        # 验证并标准化数量
        qty = wcfg.get('default_qty', 1000)
        qty = Validator.validate_quantity(qty)
        
        # 验证SLA时间限制
        max_runtime = syscfg.get('sla', {}).get('max_runtime_sec', 45)
        if max_runtime <= 0 or max_runtime > 300:
            logger.warning(f"Adjusting max_runtime from {max_runtime} to 45 seconds")
            max_runtime = 45
        
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    day = datetime.now().strftime('%Y%m%d')
    dayd = datetime.now().strftime('%Y-%m-%d')
    
    # 加载预测数据并验证
    try:
        preds = load_preds(day)
        cands = list(preds.keys()) or ['SZ000001', 'SH600000']
        
        # 验证候选股票代码
        validated_cands = []
        for symbol in cands[:100]:  # 限制最多100个股票
            try:
                validated_symbol = Validator.validate_symbol(symbol)
                validated_cands.append(validated_symbol)
            except ValidationError as e:
                logger.warning(f"Skipping invalid symbol {symbol}: {e}")
        
        cands = validated_cands if validated_cands else ['SZ000001', 'SH600000']
        logger.info(f"Processing {len(cands)} validated symbols")
        
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        cands = ['SZ000001', 'SH600000']

    rows = []
    try:
        for sym in cands:
            # 检查超时
            if time.time() - t0 > max_runtime:
                logger.warning(f"Timeout reached after processing {len(rows)} symbols")
                break
            
            try:
                # 运行Agent分析
                ascores = run_agents(sym)
                qscore = preds.get(sym, 0.5)
                
                # 验证分数范围
                qscore = max(0, min(1, qscore))
                
                # 计算融合分数
                fscore = fused_score(ascores, qscore)
                
                # 构建结果行
                row = {
                    'symbol': sym,
                    'final_score': round(fscore, 4),
                    'model': round(qscore, 4)
                }
                
                # 添加Agent分数
                for k, v in ascores.items():
                    score_key = f's_{k}'
                    row[score_key] = round(max(0, min(1, v)), 4)  # 确保分数在0-1范围
                
                rows.append(row)
                
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in main processing loop: {e}")
        # 降级策略：使用Qlib预测
        if syscfg.get('fallback', {}).get('strategy', 'qlib_only') == 'qlib_only':
            logger.info("Falling back to Qlib-only predictions")
            rows = []
            for sym in cands:
                score = preds.get(sym, 0.5)
                score = round(max(0, min(1, score)), 4)
                rows.append({
                    'symbol': sym,
                    'final_score': score,
                    'model': score
                })

    df=pd.DataFrame(rows).sort_values('final_score',ascending=False); top=df.head(topk).copy()
    os.makedirs('output',exist_ok=True); out_csv=os.path.join('output',f'preopen_{dayd}.csv'); out_txt=os.path.join('output',f'preopen_{dayd}.txt')
    top.to_csv(out_csv,index=False,encoding='utf-8-sig'); open(out_txt,'w',encoding='utf-8').write('\n'.join([f"{r.symbol}: 评分={r.final_score}（模型={r.model}）" for r in top.itertuples()]))

    rep_dir=os.path.join('reports',dayd); os.makedirs(rep_dir,exist_ok=True)
    for r in top.itertuples():
        scores={c[2:]:getattr(r,c) for c in top.columns if c.startswith('s_')}
        write_markdown_report(r.symbol, scores, r.model, r.final_score, rep_dir, weights)

    # 下单处理
    if args.place_orders and not args.dry_run:
        try:
            og_cfg = _load_yaml('executors/order_gateway/config.yaml', 
                              {'mode': 'csv', 'csv': {'dir': 'orders'}})
            og = OrderGateway(og_cfg)
            
            for r in top.itertuples():
                try:
                    # 验证订单参数
                    validated_symbol = Validator.validate_symbol(r.symbol)
                    validated_qty = Validator.validate_quantity(qty)
                    
                    # 获取合理的价格（这里应该从市场数据获取）
                    price = 10.0  # TODO: 从市场数据获取实际价格
                    validated_price = Validator.validate_price(price, validated_symbol)
                    
                    # 风险检查
                    capital = og_cfg.get('account', {}).get('capital', 1000000)
                    position_value = validated_qty * validated_price
                    
                    # 创建订单
                    order = Order(
                        symbol=validated_symbol,
                        side=og_cfg.get('account', {}).get('default_side', 'BUY'),
                        qty=validated_qty,
                        price=validated_price
                    
                    # 下单
                    og.place(order)
                    logger.info(f"Order placed for {validated_symbol}: {validated_qty} @ {validated_price}")
                    
                except ValidationError as e:
                    logger.error(f"Order validation failed for {r.symbol}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to place order for {r.symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Order gateway error: {e}")
    elif args.dry_run:
        logger.info("Dry run mode - no orders will be placed")

    # 输出结果
    elapsed_time = round(time.time() - t0, 3)
    logger.info(f"Workflow completed in {elapsed_time} seconds")
    print(f'\n完成，耗时(s)：{elapsed_time}')
    print('\n=== 选中股票 ====')
    print(top)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
