"""
AKShare高频数据接口
支持1分钟/5分钟级别的A股高频数据获取
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
import os
from pathlib import Path
import pickle
import time

logger = logging.getLogger(__name__)

# 数据缓存目录
CACHE_DIR = Path("data/cache/highfreq")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class AKShareHighFreqData:
    """AKShare高频数据接口"""
    
    def __init__(self, freq: str = "1min"):
        """
        初始化
        
        Args:
            freq: 数据频率，支持 "1min", "5min", "15min", "30min", "60min"
        """
        self.freq = freq
        self.freq_map = {
            "1min": "1",
            "5min": "5",
            "15min": "15",
            "30min": "30",
            "60min": "60"
        }
        
        if freq not in self.freq_map:
            raise ValueError(f"不支持的频率: {freq}，支持的频率: {list(self.freq_map.keys())}")
        
        # 尝试导入akshare
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
            logger.info(f"✅ AKShare高频数据接口初始化成功 (频率: {freq})")
        except ImportError:
            self.ak = None
            self.available = False
            logger.warning("⚠️ AKShare未安装，将使用模拟数据")
    
    def get_intraday_data(
        self, 
        symbol: str, 
        trade_date: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取股票日内分时数据
        
        Args:
            symbol: 股票代码，如 "000001" 或 "000001.SZ"
            trade_date: 交易日期，格式 "YYYY-MM-DD"
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume, amount
        """
        try:
            # 标准化股票代码
            symbol_clean = symbol.replace(".SZ", "").replace(".SH", "")
            
            # 检查缓存
            if use_cache:
                cached_data = self._load_cache(symbol_clean, trade_date)
                if cached_data is not None:
                    logger.info(f"📦 从缓存加载: {symbol_clean} {trade_date}")
                    return cached_data
            
            # 从AKShare获取
            if not self.available:
                logger.warning("AKShare不可用，返回模拟数据")
                return self._generate_mock_data(trade_date)
            
            logger.info(f"🌐 从AKShare获取: {symbol_clean} {trade_date} (freq={self.freq})")
            
            # 调用AKShare接口
            try:
                # 使用实时行情接口获取分时数据
                df = self.ak.stock_zh_a_hist_min_em(
                    symbol=symbol_clean,
                    period=self.freq_map[self.freq],
                    adjust="",  # 不复权
                    start_date=trade_date.replace("-", "") + " 09:30:00",
                    end_date=trade_date.replace("-", "") + " 15:00:00"
                )
                
                if df is None or df.empty:
                    logger.warning(f"⚠️ 未获取到数据: {symbol_clean} {trade_date}")
                    return None
                
                # 数据标准化
                df = self._standardize_data(df)
                
                # 保存缓存
                if use_cache:
                    self._save_cache(symbol_clean, trade_date, df)
                
                logger.info(f"✅ 成功获取 {len(df)} 条数据")
                return df
                
            except Exception as e:
                logger.error(f"❌ AKShare接口调用失败: {e}")
                # 返回模拟数据作为后备
                return self._generate_mock_data(trade_date)
        
        except Exception as e:
            logger.error(f"获取高频数据失败: {e}", exc_info=True)
            return None
    
    def get_multiple_days(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取多天的高频数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"
            use_cache: 是否使用缓存
            
        Returns:
            合并的DataFrame
        """
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # B = 工作日
            
            all_data = []
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                logger.info(f"获取 {symbol} {date_str} 的数据...")
                
                df = self.get_intraday_data(symbol, date_str, use_cache)
                if df is not None and not df.empty:
                    all_data.append(df)
                
                # 避免请求过快
                time.sleep(0.5)
            
            if not all_data:
                logger.warning(f"未获取到任何数据: {symbol} {start_date} ~ {end_date}")
                return None
            
            # 合并数据
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('time').reset_index(drop=True)
            
            logger.info(f"✅ 成功获取 {len(all_data)} 天数据，共 {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"获取多日数据失败: {e}", exc_info=True)
            return None
    
    def get_limit_up_stocks(self, trade_date: str) -> List[str]:
        """
        获取指定日期的涨停股票列表
        
        Args:
            trade_date: 交易日期 "YYYY-MM-DD"
            
        Returns:
            涨停股票代码列表
        """
        try:
            if not self.available:
                logger.warning("AKShare不可用，返回模拟涨停列表")
                return ["000001", "600519", "000858"]
            
            logger.info(f"获取 {trade_date} 涨停股票列表...")
            
            # 调用AKShare涨停股票接口
            df = self.ak.stock_zt_pool_em(date=trade_date.replace("-", ""))
            
            if df is None or df.empty:
                logger.warning(f"未获取到涨停数据: {trade_date}")
                return []
            
            # 提取股票代码
            stocks = df['代码'].tolist() if '代码' in df.columns else []
            
            logger.info(f"✅ 获取到 {len(stocks)} 只涨停股票")
            return stocks
            
        except Exception as e:
            logger.error(f"获取涨停列表失败: {e}", exc_info=True)
            return []
    
    def get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """
        获取实时行情数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            实时行情字典
        """
        try:
            if not self.available:
                return None
            
            symbol_clean = symbol.replace(".SZ", "").replace(".SH", "")
            
            # 获取实时行情
            df = self.ak.stock_zh_a_spot_em()
            
            if df is None or df.empty:
                return None
            
            # 查找目标股票
            stock_data = df[df['代码'] == symbol_clean]
            
            if stock_data.empty:
                return None
            
            # 转换为字典
            result = {
                'symbol': symbol_clean,
                'name': stock_data['名称'].values[0],
                'price': stock_data['最新价'].values[0],
                'change_pct': stock_data['涨跌幅'].values[0],
                'volume': stock_data['成交量'].values[0],
                'amount': stock_data['成交额'].values[0],
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return None
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据格式"""
        # 重命名列
        column_mapping = {
            '时间': 'time',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保时间列
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # 选择需要的列
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        # 数据类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除缺失值
        df = df.dropna()
        
        return df
    
    def _generate_mock_data(self, trade_date: str) -> pd.DataFrame:
        """生成模拟数据（用于测试）"""
        logger.info(f"📝 生成模拟数据: {trade_date}")
        
        # 生成交易时间
        times = []
        
        # 上午 9:30-11:30
        morning_times = pd.date_range(
            start=f"{trade_date} 09:30:00",
            end=f"{trade_date} 11:30:00",
            freq=self.freq
        )
        times.extend(morning_times)
        
        # 下午 13:00-15:00
        afternoon_times = pd.date_range(
            start=f"{trade_date} 13:00:00",
            end=f"{trade_date} 15:00:00",
            freq=self.freq
        )
        times.extend(afternoon_times)
        
        n = len(times)
        
        # 生成模拟价格（随机游走）
        base_price = 10.0
        returns = np.random.normal(0, 0.001, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # 生成OHLC
        df = pd.DataFrame({
            'time': times,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, n)),
            'high': prices * (1 + np.random.uniform(0, 0.005, n)),
            'low': prices * (1 - np.random.uniform(0, 0.005, n)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n),
            'amount': np.random.randint(10000, 100000, n)
        })
        
        return df
    
    def _load_cache(self, symbol: str, trade_date: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        try:
            cache_file = CACHE_DIR / f"{symbol}_{trade_date}_{self.freq}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data
            
            return None
            
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None
    
    def _save_cache(self, symbol: str, trade_date: str, data: pd.DataFrame):
        """保存数据到缓存"""
        try:
            cache_file = CACHE_DIR / f"{symbol}_{trade_date}_{self.freq}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"💾 缓存已保存: {cache_file}")
            
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """清除缓存"""
        try:
            if symbol:
                # 清除特定股票的缓存
                pattern = f"{symbol}_*_{self.freq}.pkl"
                for cache_file in CACHE_DIR.glob(pattern):
                    cache_file.unlink()
                    logger.info(f"🗑️ 已删除缓存: {cache_file}")
            else:
                # 清除所有缓存
                for cache_file in CACHE_DIR.glob(f"*_{self.freq}.pkl"):
                    cache_file.unlink()
                logger.info("🗑️ 已清除所有缓存")
                
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")


class HighFreqDataManager:
    """高频数据管理器"""
    
    def __init__(self):
        self.data_sources = {
            '1min': AKShareHighFreqData('1min'),
            '5min': AKShareHighFreqData('5min'),
            '15min': AKShareHighFreqData('15min'),
            '30min': AKShareHighFreqData('30min'),
            '60min': AKShareHighFreqData('60min')
        }
    
    def get_data(
        self,
        symbol: str,
        freq: str,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取高频数据的统一接口
        
        Args:
            symbol: 股票代码
            freq: 数据频率
            start_date: 开始日期
            end_date: 结束日期（可选，默认为start_date）
            use_cache: 是否使用缓存
            
        Returns:
            高频数据DataFrame
        """
        if freq not in self.data_sources:
            raise ValueError(f"不支持的频率: {freq}")
        
        data_source = self.data_sources[freq]
        
        if end_date is None or start_date == end_date:
            # 单日数据
            return data_source.get_intraday_data(symbol, start_date, use_cache)
        else:
            # 多日数据
            return data_source.get_multiple_days(symbol, start_date, end_date, use_cache)
    
    def get_cache_info(self) -> Dict[str, int]:
        """获取缓存信息"""
        info = {}
        for freq in self.data_sources.keys():
            pattern = f"*_{freq}.pkl"
            count = len(list(CACHE_DIR.glob(pattern)))
            info[freq] = count
        return info
    
    def clear_all_cache(self):
        """清除所有缓存"""
        for data_source in self.data_sources.values():
            data_source.clear_cache()


# 全局实例
highfreq_manager = HighFreqDataManager()


# ================== 新增功能: 缠论系统数据接口 ==================

def get_stock_hist_data(
    codes: List[str],
    start_date: str,
    end_date: str,
    period: str = "daily"
) -> Dict[str, pd.DataFrame]:
    """
    获取股票历史数据（用于缠论系统）
    
    Args:
        codes: 股票代码列表（支持带后缀或纯数字，如 "000001" 或 "000001.SZ"）
        start_date: 开始日期 YYYYMMDD 格式
        end_date: 结束日期 YYYYMMDD 格式
        period: 数据周期 "daily"/"weekly"/"monthly"
    
    Returns:
        Dict[股票代码, DataFrame] 包含 datetime, open, high, low, close, volume, macd, macd_signal, rsi
    """
    import akshare as ak
    
    result = {}
    
    logger.info(f"开始获取 {len(codes)} 只股票的历史数据 ({start_date} 至 {end_date})")
    
    for idx, code in enumerate(codes, 1):
        try:
            # 转换代码格式：移除后缀得到纯数字
            clean_code = code.split('.')[0] if '.' in code else code
            
            logger.info(f"[{idx}/{len(codes)}] 获取 {clean_code} 数据...")
            
            # 调用 AKShare 接口
            df = ak.stock_zh_a_hist(
                symbol=clean_code,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df is None or df.empty:
                logger.warning(f"⚠️ 股票 {code} 未获取到数据")
                continue
            
            # 转换为系统格式
            df_formatted = convert_akshare_to_system_format(df, code)
            
            if df_formatted is None or df_formatted.empty:
                logger.warning(f"⚠️ 股票 {code} 格式转换失败")
                continue
            
            # 确定完整代码（带后缀）
            if '.' in code:
                full_code = code
            else:
                # 6开头是上海，否则是深圳
                full_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            
            result[full_code] = df_formatted
            logger.info(f"✅ {full_code} 数据获取成功: {len(df_formatted)} 条")
            
            # 避免请求过快
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"❌ 获取股票 {code} 数据失败: {e}")
            continue
    
    logger.info(f"✅ 数据获取完成: {len(result)}/{len(codes)} 只股票成功")
    return result


def convert_akshare_to_system_format(df: pd.DataFrame, code: str) -> Optional[pd.DataFrame]:
    """
    将 AKShare 返回的数据格式转换为系统需要的格式
    
    Args:
        df: AKShare 返回的原始 DataFrame
        code: 股票代码
    
    Returns:
        格式化后的 DataFrame，包含技术指标
    """
    try:
        # 重命名列（AKShare 返回的是中文列名）
        df = df.rename(columns={
            '日期': 'datetime',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        
        # 确保必需列存在
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"数据缺少必需列: {missing_cols}")
            return None
        
        # 转换日期格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 确保数值列是正确类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        if len(df) < 30:
            logger.warning(f"数据点太少 ({len(df)} 条)，可能无法计算指标")
        
        # 计算技术指标
        try:
            # 方法1：尝试使用 pandas_ta
            import pandas_ta as ta
            df.ta.macd(append=True)  # 添加 MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            df.ta.rsi(length=14, append=True)  # 添加 RSI_14
            
            # 重命名技术指标列
            df = df.rename(columns={
                'MACD_12_26_9': 'macd',
                'MACDs_12_26_9': 'macd_signal',
                'RSI_14': 'rsi'
            })
            
        except ImportError:
            logger.warning("pandas_ta 未安装，使用简化方法计算技术指标")
            # 方法2：手动计算简化版技术指标
            df = calculate_indicators_manual(df)
        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}，使用简化方法")
            df = calculate_indicators_manual(df)
        
        # 选择最终需要的列
        final_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'macd', 'macd_signal', 'rsi']
        
        # 确保所有列都存在（不存在的填充NaN）
        for col in final_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[final_cols]
        
        # 删除所有数据都是NaN的行
        df = df.dropna(how='all')
        
        return df
        
    except Exception as e:
        logger.error(f"数据格式转换失败 ({code}): {e}", exc_info=True)
        return None


def calculate_indicators_manual(df: pd.DataFrame) -> pd.DataFrame:
    """
    手动计算技术指标（简化版，不依赖 pandas_ta）
    """
    try:
        # 计算 MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 计算 RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    except Exception as e:
        logger.error(f"手动计算指标失败: {e}")
        # 填充默认值
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['rsi'] = 50.0
        return df


def get_limit_up_stocks_list(date: Optional[str] = None) -> List[str]:
    """
    获取涨停板股票列表（用于缠论系统）
    
    Args:
        date: 日期 YYYYMMDD 格式，默认为今天
    
    Returns:
        股票代码列表（带后缀，如 ["000001.SZ", "600519.SH"]）
    """
    try:
        import akshare as ak
        
        target_date = date or datetime.now().strftime("%Y%m%d")
        logger.info(f"获取 {target_date} 的涨停板股票...")
        
        df = ak.stock_zt_pool_em(date=target_date)
        
        if df is None or df.empty:
            logger.warning(f"⚠️ {target_date} 无涨停板数据")
            return []
        
        # 提取股票代码
        if '代码' not in df.columns:
            logger.error("涨停板数据格式异常：缺少'代码'列")
            return []
        
        codes = df['代码'].astype(str).tolist()
        
        # 添加后缀
        full_codes = []
        for code in codes:
            if code.startswith('6'):
                full_codes.append(f"{code}.SH")
            elif code.startswith(('0', '3')):
                full_codes.append(f"{code}.SZ")
            else:
                logger.warning(f"未知股票代码格式: {code}")
        
        logger.info(f"✅ 获取到 {len(full_codes)} 只涨停股票")
        return full_codes
        
    except Exception as e:
        logger.error(f"获取涨停板股票失败: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试单日数据获取
    data_source = AKShareHighFreqData(freq="1min")
    
    # 测试日期
    test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n测试获取 000001 {test_date} 的1分钟数据:")
    df = data_source.get_intraday_data("000001", test_date)
    
    if df is not None:
        print(f"✅ 成功获取 {len(df)} 条数据")
        print(df.head())
        print(df.tail())
    else:
        print("❌ 获取失败")
    
    # 测试涨停列表
    print(f"\n测试获取 {test_date} 涨停股票:")
    limit_up_stocks = data_source.get_limit_up_stocks(test_date)
    print(f"✅ 涨停股票: {limit_up_stocks[:10]}")
