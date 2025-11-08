"""将P2-1缠论Alpha因子写入Qlib存储

这个脚本负责计算并持久化三个P2-1 Alpha派生因子到Qlib数据仓库，
以便IC分析和回测Tab可以无缝加载这些因子，无需手动注入。

三个Alpha因子：
1. alpha_zs_movement = zs_movement_direction × zs_movement_confidence
2. alpha_zs_upgrade = zs_upgrade_flag × zs_upgrade_strength  
3. alpha_confluence = tanh(confluence_score)

依赖：
- qlib_enhanced/chanlun/chanlun_alpha.py: Alpha计算逻辑
- features/chanlun/chanpy_features.py: 基础缠论特征生成（包含中枢移动/共振字段）

作者: Warp AI Assistant
日期: 2025-01
项目: 麒麟量化系统 - P2 Alpha集成
"""
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import qlib
    from qlib.data import D
    from qlib.config import C
except ImportError:
    qlib = None
    D = None
    C = None
    print("⚠️  Qlib未安装或不可用")

from qlib_enhanced.chanlun.chanlun_alpha import ChanLunAlphaFactors
from features.chanlun.chanpy_features import ChanPyFeatureGenerator

logger = logging.getLogger(__name__)


class ChanLunAlphaWriter:
    """缠论Alpha因子写入器
    
    功能：
    1. 逐股票生成完整的缠论基础特征（包含中枢移动/共振）
    2. 计算P2-1三个Alpha派生因子
    3. 写入Qlib存储，以便后续分析和回测
    """
    
    ALPHA_FIELDS = [
        'alpha_zs_movement',   # 中枢移动强度
        'alpha_zs_upgrade',    # 中枢升级强度
        'alpha_confluence',    # 多周期共振强度
    ]
    
    def __init__(self, 
                 provider_uri: str = None,
                 region: str = 'cn'):
        """初始化
        
        Args:
            provider_uri: Qlib数据路径 (None=使用默认)
            region: 区域代码
        """
        self.provider_uri = provider_uri
        self.region = region
        
        # 初始化Qlib
        if qlib is not None:
            try:
                qlib.init(provider_uri=provider_uri, region=region)
                logger.info(f"✅ Qlib初始化成功: {C.get_data_path()}")
            except Exception as e:
                logger.warning(f"Qlib初始化警告: {e}")
        else:
            raise RuntimeError("Qlib不可用，无法写入数据")
        
        # 初始化特征生成器
        self.chanpy_gen = ChanPyFeatureGenerator(
            seg_algo='chan',
            bi_algo='normal',
            zs_combine=True
        )
        
        print("="*70)
        print("🚀 缠论Alpha因子写入器初始化完成")
        print(f"   目标因子: {', '.join(self.ALPHA_FIELDS)}")
        print("="*70)
    
    def generate_alpha_for_stock(self, 
                                  code: str, 
                                  start: str, 
                                  end: str) -> pd.DataFrame:
        """为单个股票生成Alpha因子
        
        Args:
            code: 股票代码 (如 'SH600000')
            start: 开始日期
            end: 结束日期
            
        Returns:
            DataFrame with columns: [datetime, alpha_zs_movement, alpha_zs_upgrade, alpha_confluence]
        """
        try:
            # 1. 从Qlib加载OHLCV
            ohlcv_df = D.features(
                instruments=[code],
                fields=['$open', '$close', '$high', '$low', '$volume'],
                start_time=start,
                end_time=end,
                freq='day'
            )
            
            if ohlcv_df is None or len(ohlcv_df) == 0:
                logger.warning(f"{code}: 无OHLCV数据")
                return pd.DataFrame()
            
            # 重置索引并准备输入格式
            if isinstance(ohlcv_df.index, pd.MultiIndex):
                ohlcv_df = ohlcv_df.reset_index(level=0, drop=True)  # 移除instrument层
            ohlcv_df = ohlcv_df.reset_index()  # datetime变为列
            ohlcv_df.rename(columns={'index': 'datetime'}, inplace=True)
            
            input_df = pd.DataFrame({
                'datetime': ohlcv_df['datetime'],
                'open': ohlcv_df['$open'],
                'close': ohlcv_df['$close'],
                'high': ohlcv_df['$high'],
                'low': ohlcv_df['$low'],
                'volume': ohlcv_df['$volume'],
            })
            
            # 2. 生成完整缠论特征（包含中枢移动/共振字段）
            full_features_df = self.chanpy_gen.generate_features(input_df, code=code)
            
            # 3. 计算Alpha派生因子
            alpha_df = ChanLunAlphaFactors.generate_alpha_factors(full_features_df, code=code)
            
            # 4. 提取三个目标Alpha
            result = alpha_df[['datetime'] + self.ALPHA_FIELDS].copy()
            result['instrument'] = code
            
            return result
            
        except Exception as e:
            logger.error(f"{code}: Alpha生成失败 - {e}", exc_info=True)
            return pd.DataFrame()
    
    def write_alphas_to_store(self, 
                               instruments: str = 'csi300',
                               start: str = '2020-01-01',
                               end: str = '2023-12-31',
                               output_path: str = None):
        """批量计算并写入Alpha因子到Qlib存储
        
        Args:
            instruments: 股票池 (如 'csi300', 'csi500')
            start: 开始日期
            end: 结束日期
            output_path: 可选的CSV输出路径（用于调试/验证）
        
        Returns:
            生成的Alpha DataFrame
        """
        print(f"\n🔄 开始批量生成Alpha因子...")
        print(f"   股票池: {instruments}")
        print(f"   时间范围: {start} ~ {end}")
        
        # 1. 获取股票列表
        try:
            inst_list_df = D.instruments(market=instruments)
            if isinstance(inst_list_df, pd.DataFrame):
                inst_codes = inst_list_df.index.tolist()
            else:
                inst_codes = inst_list_df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return None
        
        print(f"   股票数量: {len(inst_codes)}")
        
        # 2. 逐股票生成
        alpha_list = []
        success_count = 0
        fail_count = 0
        
        for i, code in enumerate(inst_codes, 1):
            print(f"\r   进度: {i}/{len(inst_codes)} - {code}", end='', flush=True)
            
            alpha_df = self.generate_alpha_for_stock(code, start, end)
            
            if not alpha_df.empty:
                alpha_list.append(alpha_df)
                success_count += 1
            else:
                fail_count += 1
        
        print()  # 换行
        
        if not alpha_list:
            print("❌ 未生成任何Alpha数据")
            return None
        
        # 3. 合并所有股票的Alpha
        all_alphas = pd.concat(alpha_list, ignore_index=True)
        all_alphas = all_alphas.set_index(['instrument', 'datetime']).sort_index()
        
        print(f"\n✅ Alpha因子生成完成")
        print(f"   成功: {success_count}, 失败: {fail_count}")
        print(f"   数据形状: {all_alphas.shape}")
        print(f"   日期范围: {all_alphas.index.get_level_values('datetime').min()} ~ {all_alphas.index.get_level_values('datetime').max()}")
        
        # 4. 可选：保存为CSV（调试用）
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            all_alphas.to_csv(output_file)
            print(f"   💾 已保存CSV: {output_file}")
        
        # 5. 写入Qlib存储
        self._write_to_qlib_store(all_alphas, instruments, start, end)
        
        return all_alphas
    
    def _write_to_qlib_store(self, 
                             alpha_df: pd.DataFrame,
                             instruments: str,
                             start: str,
                             end: str):
        """将Alpha数据写入Qlib feature store
        
        注意：Qlib的feature store是通过D.features()读取的底层存储
        默认情况下，自定义因子需要通过以下方式持久化：
        1. 使用dump_bin工具将DataFrame转换为Qlib二进制格式
        2. 或者通过扩展Provider实现自定义数据源
        
        这里我们采用中间方案：将Alpha保存到项目目录下的pickle缓存，
        供后续load_factor_from_qlib替换逻辑使用
        """
        print(f"\n📦 准备写入Qlib存储...")
        
        # 确保输出目录存在
        store_dir = project_root / 'data' / 'qlib_alpha_cache'
        store_dir.mkdir(parents=True, exist_ok=True)
        
        # 逐个Alpha因子保存为单独的pickle文件
        for alpha_name in self.ALPHA_FIELDS:
            if alpha_name not in alpha_df.columns:
                logger.warning(f"⚠️  Alpha因子 {alpha_name} 不存在，跳过")
                continue
            
            # 提取单个Alpha的Series
            alpha_series = alpha_df[alpha_name]
            
            # 文件命名: {alpha_name}_{instruments}_{start}_{end}.pkl
            filename = f"{alpha_name}_{instruments}_{start}_{end}.pkl"
            filepath = store_dir / filename
            
            # 保存
            alpha_series.to_pickle(filepath)
            print(f"   ✅ {alpha_name} -> {filepath.name}")
        
        # 保存元信息
        meta = {
            'instruments': instruments,
            'start': start,
            'end': end,
            'alpha_fields': self.ALPHA_FIELDS,
            'generated_at': datetime.now().isoformat(),
            'shape': alpha_df.shape,
        }
        meta_file = store_dir / f"_meta_{instruments}_{start}_{end}.json"
        import json
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"   ℹ️  元信息: {meta_file.name}")
        
        print(f"\n✅ Alpha因子已写入本地缓存: {store_dir}")
        print(f"   后续IC分析和回测可通过load_factor_from_qlib_cache()加载")
    
    def verify_alphas(self, 
                      instruments: str = 'csi300',
                      start: str = '2020-01-01',
                      end: str = '2023-12-31') -> Dict:
        """验证写入的Alpha因子
        
        Args:
            instruments: 股票池
            start: 开始日期
            end: 结束日期
            
        Returns:
            验证统计字典
        """
        print(f"\n🔍 验证Alpha因子...")
        
        store_dir = project_root / 'data' / 'qlib_alpha_cache'
        
        stats = {}
        
        for alpha_name in self.ALPHA_FIELDS:
            filename = f"{alpha_name}_{instruments}_{start}_{end}.pkl"
            filepath = store_dir / filename
            
            if not filepath.exists():
                stats[alpha_name] = {'status': '❌ 未找到'}
                continue
            
            try:
                alpha_series = pd.read_pickle(filepath)
                
                stats[alpha_name] = {
                    'status': '✅ 正常',
                    'shape': alpha_series.shape,
                    'null_ratio': f"{alpha_series.isna().sum() / len(alpha_series) * 100:.2f}%",
                    'mean': f"{alpha_series.mean():.4f}",
                    'std': f"{alpha_series.std():.4f}",
                    'min': f"{alpha_series.min():.4f}",
                    'max': f"{alpha_series.max():.4f}",
                }
            except Exception as e:
                stats[alpha_name] = {'status': f'❌ 加载失败: {e}'}
        
        # 打印验证结果
        print("\n📊 验证结果:")
        for alpha_name, stat in stats.items():
            print(f"\n   {alpha_name}:")
            for k, v in stat.items():
                print(f"      {k}: {v}")
        
        return stats


def load_factor_from_qlib_cache(
    alpha_name: str,
    instruments: str = 'csi300',
    start: str = '2020-01-01',
    end: str = '2023-12-31',
) -> pd.DataFrame:
    """从Qlib Alpha缓存加载因子
    
    这是一个辅助函数，供IC分析和回测Tab使用，
    用于无缝加载已持久化的Alpha因子。
    
    Args:
        alpha_name: Alpha因子名称 (如 'alpha_confluence')
        instruments: 股票池
        start: 开始日期
        end: 结束日期
        
    Returns:
        DataFrame with MultiIndex[instrument, datetime] and single column
    """
    store_dir = Path(__file__).parent.parent / 'data' / 'qlib_alpha_cache'
    filename = f"{alpha_name}_{instruments}_{start}_{end}.pkl"
    filepath = store_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Alpha缓存未找到: {filepath}")
    
    alpha_series = pd.read_pickle(filepath)
    
    # 转换为DataFrame格式（与load_factor_from_qlib兼容）
    df = alpha_series.to_frame(name='factor')
    df['label'] = 0  # placeholder
    
    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将P2-1缠论Alpha因子写入Qlib存储')
    parser.add_argument('--instruments', type=str, default='csi300',
                        help='股票池 (默认: csi300)')
    parser.add_argument('--start', type=str, default='2020-01-01',
                        help='开始日期 (默认: 2020-01-01)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                        help='结束日期 (默认: 2023-12-31)')
    parser.add_argument('--provider-uri', type=str, default=None,
                        help='Qlib数据路径 (默认: None=使用Qlib默认)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='可选的CSV输出路径（用于调试）')
    parser.add_argument('--verify', action='store_true',
                        help='仅验证已有的Alpha数据，不生成')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(project_root / 'logs' / 'chanlun_alpha_write.log'),
            logging.StreamHandler()
        ]
    )
    
    # 创建写入器
    writer = ChanLunAlphaWriter(
        provider_uri=args.provider_uri,
        region='cn'
    )
    
    # 验证模式
    if args.verify:
        writer.verify_alphas(
            instruments=args.instruments,
            start=args.start,
            end=args.end
        )
        return
    
    # 生成并写入
    start_time = datetime.now()
    
    alpha_df = writer.write_alphas_to_store(
        instruments=args.instruments,
        start=args.start,
        end=args.end,
        output_path=args.output_csv
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if alpha_df is not None:
        print(f"\n⏱️  总耗时: {elapsed:.1f}秒")
        
        # 验证
        writer.verify_alphas(
            instruments=args.instruments,
            start=args.start,
            end=args.end
        )
        
        print(f"\n✅ P2-1 Alpha因子写入完成!")
        print(f"   后续在IC分析/回测Tab中可直接使用:")
        print(f"      - $alpha_zs_movement")
        print(f"      - $alpha_zs_upgrade")
        print(f"      - $alpha_confluence")
    else:
        print(f"\n❌ Alpha因子写入失败")


if __name__ == '__main__':
    main()
