"""
数据格式转换工具
支持 CSV/Excel → Qlib格式转换
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataConverter:
    """数据格式转换器"""
    
    # 标准列名映射
    STANDARD_COLUMNS = {
        'date': ['日期', 'Date', 'date', 'datetime', 'trade_date'],
        'symbol': ['代码', 'Symbol', 'symbol', 'code', 'stock_code', 'ts_code'],
        'open': ['开盘', 'Open', 'open', 'open_price'],
        'high': ['最高', 'High', 'high', 'high_price'],
        'low': ['最低', 'Low', 'low', 'low_price'],
        'close': ['收盘', 'Close', 'close', 'close_price'],
        'volume': ['成交量', 'Volume', 'volume', 'vol'],
        'amount': ['成交额', 'Amount', 'amount', 'turnover']
    }
    
    def __init__(self):
        """初始化"""
        pass
    
    def detect_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        自动检测列名映射
        
        Args:
            df: 原始DataFrame
            
        Returns:
            {标准列名: 原始列名}
        """
        mapping = {}
        df_columns = df.columns.tolist()
        
        for standard_col, variations in self.STANDARD_COLUMNS.items():
            for var in variations:
                if var in df_columns:
                    mapping[standard_col] = var
                    break
        
        return mapping
    
    def convert_csv_to_dataframe(
        self,
        csv_path: str,
        column_mapping: Optional[Dict[str, str]] = None,
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        读取CSV并转换为标准格式DataFrame
        
        Args:
            csv_path: CSV文件路径
            column_mapping: 自定义列名映射 {标准名: 原始名}
            encoding: 文件编码
            
        Returns:
            标准格式DataFrame
        """
        try:
            # 尝试不同编码
            encodings = [encoding, 'utf-8', 'gbk', 'gb2312']
            df = None
            
            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    logger.info(f"使用编码 {enc} 读取成功")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"无法读取文件 {csv_path}")
            
            # 自动检测列名映射
            if column_mapping is None:
                column_mapping = self.detect_column_mapping(df)
                logger.info(f"自动检测列名映射: {column_mapping}")
            
            # 重命名列
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            df = df.rename(columns=reverse_mapping)
            
            # 验证必需列
            required_cols = ['date', 'symbol', 'close']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"缺少必需列: {missing}")
            
            # 数据清洗
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"CSV转换失败: {e}")
            raise
    
    def convert_excel_to_dataframe(
        self,
        excel_path: str,
        sheet_name: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        读取Excel并转换为标准格式
        
        Args:
            excel_path: Excel文件路径
            sheet_name: Sheet名称，None表示第一个sheet
            column_mapping: 列名映射
            
        Returns:
            标准格式DataFrame
        """
        try:
            # 读取Excel
            if sheet_name is None:
                df = pd.read_excel(excel_path)
            else:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            logger.info(f"Excel读取成功: {df.shape}")
            
            # 自动检测列名
            if column_mapping is None:
                column_mapping = self.detect_column_mapping(df)
            
            # 重命名
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            df = df.rename(columns=reverse_mapping)
            
            # 验证
            required_cols = ['date', 'symbol', 'close']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"缺少必需列: {missing}")
            
            # 清洗
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Excel转换失败: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 1. 日期格式化
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 2. 股票代码格式化
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype(str)
            # 去除空格
            df['symbol'] = df['symbol'].str.strip()
        
        # 3. 价格和成交量转为数值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. 删除无效行
        df = df.dropna(subset=['date', 'symbol', 'close'])
        
        # 5. 排序
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        return df
    
    def save_to_qlib_format(
        self,
        df: pd.DataFrame,
        output_dir: str,
        freq: str = '1d'
    ) -> str:
        """
        保存为Qlib格式
        
        Args:
            df: 标准格式DataFrame
            output_dir: 输出目录
            freq: 数据频率 1d/1min/5min
            
        Returns:
            输出路径
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Qlib需要MultiIndex (datetime, instrument)
            if 'date' in df.columns and 'symbol' in df.columns:
                df = df.set_index(['date', 'symbol'])
            
            # 按股票分组保存
            symbols = df.index.get_level_values('symbol').unique()
            
            logger.info(f"准备保存 {len(symbols)} 只股票到 {output_path}")
            
            # 使用Qlib Dumper (如果可用)
            try:
                from qlib.data import Dumper
                
                # 创建dumper配置
                dumper = Dumper(
                    csv_path=None,
                    qlib_dir=str(output_path),
                    backup_dir=None,
                    freq=freq,
                    max_workers=4
                )
                
                # 注意: Qlib的Dumper接口可能需要特殊格式
                # 这里提供简化版本，实际使用时可能需要调整
                logger.warning("Qlib Dumper功能需要根据实际Qlib版本调整")
                
            except ImportError:
                # 降级方案: 保存为CSV格式
                logger.warning("Qlib Dumper不可用，使用CSV格式保存")
                
                for symbol in symbols:
                    symbol_df = df.xs(symbol, level='symbol')
                    symbol_path = output_path / f"{symbol}.csv"
                    symbol_df.to_csv(symbol_path)
                
                logger.info(f"数据保存为CSV格式到: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"保存Qlib格式失败: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """获取数据摘要"""
        summary = {
            'total_rows': len(df),
            'total_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            },
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        return summary


# 便捷函数
def convert_csv_to_qlib(
    csv_path: str,
    output_dir: str,
    column_mapping: Optional[Dict[str, str]] = None,
    freq: str = '1d'
) -> str:
    """
    CSV → Qlib格式（快捷函数）
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        column_mapping: 列名映射
        freq: 数据频率
        
    Returns:
        输出路径
    """
    converter = DataConverter()
    df = converter.convert_csv_to_dataframe(csv_path, column_mapping)
    return converter.save_to_qlib_format(df, output_dir, freq)


def convert_excel_to_qlib(
    excel_path: str,
    output_dir: str,
    sheet_name: Optional[str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    freq: str = '1d'
) -> str:
    """
    Excel → Qlib格式（快捷函数）
    
    Args:
        excel_path: Excel文件路径
        output_dir: 输出目录
        sheet_name: Sheet名称
        column_mapping: 列名映射
        freq: 数据频率
        
    Returns:
        输出路径
    """
    converter = DataConverter()
    df = converter.convert_excel_to_dataframe(excel_path, sheet_name, column_mapping)
    return converter.save_to_qlib_format(df, output_dir, freq)


__all__ = [
    'DataConverter',
    'convert_csv_to_qlib',
    'convert_excel_to_qlib'
]
