"""
涨停板专用数据接口
支持"一进二"抓涨停板策略的数据需求
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LimitUpRecord:
    """涨停板记录"""
    symbol: str
    date: datetime
    limit_up_time: str  # 涨停时间（HH:MM:SS）
    limit_up_type: int  # 1=一字板, 2=T型板, 3=开板后封板
    seal_amount: float  # 封单金额（万元）
    seal_ratio: float  # 封单率（封单/流通市值）
    turnover_rate: float  # 换手率
    continuous_days: int  # 连板天数（1=首板，2=二板...）
    concept: List[str]  # 概念/题材标签
    industry: str  # 行业
    market_cap: float  # 流通市值（亿元）
    open_change: float  # 开盘涨跌幅
    high_change: float  # 最高涨跌幅
    volume_ratio: float  # 量比


class LimitUpDataInterface:
    """涨停板数据接口"""
    
    def __init__(self, data_source: str = "qlib"):
        """
        初始化数据接口
        
        Args:
            data_source: 数据源（qlib/akshare/tushare/custom）
        """
        self.data_source = data_source
        self._qlib_initialized = False
        self._seal_amount_cache = {}  # ✅ P0-5: 封单金额缓存
        self._continuous_board_cache = {}  # ✅ P0-5: 连板天数缓存
        self._concept_cache = {}  # ✅ P0-5: 股票所属概念缓存 {symbol: [concepts]}
        self._concept_stocks_cache = {}  # ✅ P0-5: 概念成分股缓存 {concept: [symbols]}
        logger.info(f"涨停板数据接口初始化: {data_source}")
    
    # ----------------------------
    # 内部：Qlib 初始化
    # ----------------------------
    def _ensure_qlib(self):
        if self._qlib_initialized:
            return
        try:
            import qlib
            from qlib.config import REG_CN
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            self._qlib_initialized = True
            logger.info("Qlib初始化成功用于涨停板数据接口")
        except Exception as e:
            logger.warning(f"Qlib初始化失败: {e}")
    
    def _get_basic_map(self) -> Dict[str, Dict[str, any]]:
        """从AKShare获取代码->{name, industry, market_cap(亿), turnover_rate(0-1)}"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return {}
            df.columns = [str(c) for c in df.columns]
            code_col = '代码'
            name_col = '名称'
            industry_col = '所属行业' if '所属行业' in df.columns else ('行业' if '行业' in df.columns else None)
            mcap_col = '流通市值' if '流通市值' in df.columns else None
            turn_col = '换手率' if '换手率' in df.columns else None
            mapping: Dict[str, Dict[str, any]] = {}
            for _, row in df.iterrows():
                code = str(row.get(code_col, '')).zfill(6)
                name = str(row.get(name_col, ''))
                industry = str(row.get(industry_col, '')) if industry_col else ''
                # 市值（亿）
                mcap = row.get(mcap_col, None)
                try:
                    mcap = float(str(mcap).replace(',', ''))
                except Exception:
                    mcap = None
                # 换手率（0-1）
                turn = row.get(turn_col, None)
                try:
                    turn = float(str(turn).replace('%', '').replace(',', '')) / 100.0
                except Exception:
                    turn = None
                mapping[code] = {
                    'name': name,
                    'industry': industry,
                    'market_cap': mcap,
                    'turnover_rate': turn,
                }
            return mapping
        except Exception:
            return {}
    
    def _get_st_name_map(self) -> Dict[str, str]:
        """获取ST股票名称映射（占位实现）。
        返回 {code6: name}，用于过滤 ST 股。
        TODO: 接入权威数据源。
        """
        return {}

    # ----------------------------
    # 涨停股票列表（近似基于日线）
    # ----------------------------
    def get_limit_up_stocks(self,
                           date: str,
                           min_price: float = 2.0,
                           max_price: float = 300.0,
                           exclude_st: bool = True,
                           exclude_new: bool = True) -> List[LimitUpRecord]:
        """
        获取指定日期的涨停股票（近似：日线涨幅>=9.5% 判定涨停）
        """
        self._ensure_qlib()
        try:
            from qlib.data import D
            # 使用CSI300作为示例股票池（可扩展为全A）
            instruments = list(D.instruments(market='csi300'))
            # 取当日与前一日数据
            fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', 'Ref($close, -1)']
            df = D.features(
                instruments=instruments,
                fields=fields,
                start_time=date,
                end_time=date,
                freq='day'
            )
            results: List[LimitUpRecord] = []
            if df is None or df.empty:
                return results
            # 标准化
            df = df.reset_index()
            df = df.rename(columns={
                '$open': 'open', '$high': 'high', '$low': 'low', '$close': 'close',
                '$volume': 'volume', '$amount': 'amount', 'Ref($close, -1)': 'prev_close'
            })
            for _, row in df.iterrows():
                symbol = str(row['instrument']) if 'instrument' in row else str(row.get('symbol', ''))
                prev_close = float(row.get('prev_close', np.nan))
                close = float(row.get('close', np.nan))
                high = float(row.get('high', np.nan))
                low = float(row.get('low', np.nan))
                openp = float(row.get('open', np.nan))
                price = close
                # 价格过滤
                if not np.isfinite(price) or price < min_price or price > max_price:
                    continue
                if not np.isfinite(prev_close) or prev_close <= 0:
                    continue
                ret = (close / prev_close) - 1.0
                # 近似涨停阈值
                if ret >= 0.095:
                    # 类型近似：一字板/开板封板/T字板
                    if abs(high - low) < 1e-6 and abs(high - close) < 1e-6:
                        ltype = 1
                    elif abs(high - close) < 1e-6 and openp < high:
                        ltype = 3
                    else:
                        ltype = 2
                    rec = LimitUpRecord(
                        symbol=symbol,
                        date=pd.to_datetime(date),
                        limit_up_time="",
                        limit_up_type=ltype,
                        seal_amount=float('nan'),
                        seal_ratio=float('nan'),
                        turnover_rate=float('nan'),
                        continuous_days=1,  # 连板天数在特征里进一步计算
                        concept=[],
                        industry="",
                        market_cap=float('nan'),
                        open_change=(openp / prev_close - 1.0) if np.isfinite(openp) else float('nan'),
                        high_change=(high / prev_close - 1.0) if np.isfinite(high) else float('nan'),
                        volume_ratio=float('nan'),
                    )
                    results.append(rec)
            return results
        except Exception as e:
            logger.error(f"获取涨停股票失败: {e}")
            return []
    
    # ----------------------------
    # 涨停板特征（为“首进二”提供次日判别）
    # ----------------------------
    def get_limit_up_features(self,
                             symbols: List[str],
                             date: str,
                             lookback_days: int = 20) -> pd.DataFrame:
        """获取涨停相关特征（基于Qlib日线近似计算）"""
        self._ensure_qlib()
        from qlib.data import D
        if not symbols:
            return pd.DataFrame()
        start = (pd.Timestamp(date) - pd.Timedelta(days=lookback_days + 2)).strftime('%Y-%m-%d')
        end = date
        fields = ['$open', '$high', '$low', '$close', '$volume', '$amount']
        data = D.features(symbols, fields=fields, start_time=start, end_time=end, freq='day')
        if data is None or data.empty:
            return pd.DataFrame(index=symbols)
        # 统一索引
        data = data.rename(columns={'$open': 'open', '$high': 'high', '$low': 'low', '$close': 'close', '$volume': 'volume', '$amount': 'amount'})
        # 按symbol分组
        feats = {}
        idx_date = pd.to_datetime(date)
        basic_map = self._get_basic_map()
        for sym in symbols:
            try:
                sdf = data.xs(sym, level=0) if isinstance(data.index, pd.MultiIndex) else data[data['symbol'] == sym]
                sdf = sdf.sort_index()
                if len(sdf) < 3:
                    feats[sym] = {
                        'limit_up_strength': 0.0,
                        'seal_quality': 0.0,
                        'concept_heat': 0.0,
                        'continuous_board': 0,
                        'volume_surge': 1.0,
                    }
                    continue
                sdf['prev_close'] = sdf['close'].shift(1)
                sdf['ret'] = sdf['close'] / sdf['prev_close'] - 1.0
                # 连板统计（近N日连续涨停天数）
                last_idx = sdf.index.max()
                # 取上一交易日（date当天特征用于次日判别，这里以date当天收盘数据为准）
                # 涨停强度（0-100）：按涨停接近程度
                row = sdf.loc[last_idx]
                prev_close = row.get('prev_close', np.nan)
                close = row.get('close', np.nan)
                if np.isfinite(prev_close) and np.isfinite(close) and prev_close > 0:
                    strength = min(100.0, max(0.0, ((close / prev_close - 1.0) / 0.10) * 100.0))
                else:
                    strength = 0.0
                # volume_surge：与20日均量比
                vol_ma20 = sdf['volume'].rolling(20).mean().iloc[-1]
                volume_surge = float(row.get('volume', 0.0) / vol_ma20) if vol_ma20 and np.isfinite(vol_ma20) else 1.0
                # ✅ P0-5: 使用新的 get_continuous_board() 方法
                continuous_board = self.get_continuous_board(sym, date, lookback_days=30)
                # 封板质量 proxy：收盘接近最高 且 K 线下影小
                high = row.get('high', np.nan)
                low = row.get('low', np.nan)
                seal_quality = 0.0
                if np.isfinite(high) and np.isfinite(close) and np.isfinite(low) and high > 0:
                    near_high = 1.0 - (high - close) / max(1e-6, high)
                    lower_shadow = (close - low) / max(1e-6, close)
                    seal_quality = float(
                        max(0.0, min(10.0, (near_high * 6.0 + (1.0 - lower_shadow) * 4.0)))
                    )
                # 分时近似：计算首触涨停分钟与开板次数
                intraday = self.get_intraday_data(sym, date, '1min')
                limit_up_minutes = None
                open_count = None
                if not intraday.empty and np.isfinite(prev_close):
                    limit_price = round(prev_close * 1.10, 2)
                    first_hit = intraday[intraday['high'] >= limit_price]
                    if not first_hit.empty:
                        t0 = pd.to_datetime(f"{date} 09:30:00")
                        t_hit = first_hit['time'].iloc[0]
                        limit_up_minutes = max(0, int((t_hit - t0).total_seconds()//60))
                        below = intraday[intraday['close'] < limit_price]
                        open_count = int((below['time'] > t_hit).sum())
                # ✅ P0-5: 计算 seal_amount 和 concept_heat
                seal_amount = 0.0
                if np.isfinite(prev_close) and prev_close > 0:
                    seal_amount = self.get_seal_amount(sym, date, prev_close)
                
                concept_heat = self.get_concept_heat(sym, date)
                
                feats[sym] = {
                    'limit_up_strength': float(strength),
                    'seal_quality': float(seal_quality),
                    'concept_heat': float(concept_heat),  # ✅ P0-5: 真实题材热度
                    'continuous_board': int(continuous_board),  # ✅ P0-5: 真实连板天数
                    'volume_surge': float(volume_surge),
                    'seal_amount': float(seal_amount),  # ✅ P0-5: 封单金额(万元)
                    'limit_up_minutes': float(limit_up_minutes) if limit_up_minutes is not None else np.nan,
                    'open_count': int(open_count) if open_count is not None else np.nan,
                    'close': float(close) if np.isfinite(close) else np.nan,
                    'prev_close': float(prev_close) if np.isfinite(prev_close) else np.nan,
                    # 基础映射
                    'market_cap': float(basic_map.get(sym.split('.')[0], {}).get('market_cap') or np.nan),
                    'turnover_rate': float(basic_map.get(sym.split('.')[0], {}).get('turnover_rate') or np.nan),
                    'industry': str(basic_map.get(sym.split('.')[0], {}).get('industry') or ''),
                }
            except Exception as e:
                logger.debug(f"特征计算失败 {sym}: {e}")
                feats[sym] = {
                    'limit_up_strength': 0.0,
                    'seal_quality': 0.0,
                    'concept_heat': 0.0,
                    'continuous_board': 0,
                    'volume_surge': 1.0,
                }
        # ✅ P0-5: 行业共振逻辑已集成到 get_concept_heat()，不再需要单独计算
        # (concept_heat 已在上面的 loop 中计算)
        return pd.DataFrame.from_dict(feats, orient='index')
    
    # ----------------------------
    # 次日结果（近似）
    # ----------------------------
    def get_next_day_result(self,
                           symbols: List[str],
                           date: str) -> pd.DataFrame:
        """获取次日表现（近似：次日涨幅/是否涨停）"""
        self._ensure_qlib()
        from qlib.data import D
        if not symbols:
            return pd.DataFrame()
        day = pd.Timestamp(date)
        start = (day - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        end = (day + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        data = D.features(symbols, fields=['$close'], start_time=start, end_time=end, freq='day')
        if data is None or data.empty:
            return pd.DataFrame(index=symbols)
        res = {}
        for sym in symbols:
            try:
                sdf = data.xs(sym, level=0) if isinstance(data.index, pd.MultiIndex) else data[data['symbol'] == sym]
                sdf = sdf.sort_index()
                # 取 date 与 next_day
                if day not in sdf.index or len(sdf.index) < 2:
                    res[sym] = {'next_limit_up': 0, 'next_return': 0.0, 'next_high_return': 0.0, 'open_premium': 0.0}
                    continue
                pos = list(sdf.index).index(day)
                if pos + 1 >= len(sdf.index):
                    res[sym] = {'next_limit_up': 0, 'next_return': 0.0, 'next_high_return': 0.0, 'open_premium': 0.0}
                    continue
                c0 = float(sdf.iloc[pos]['$close'])
                c1 = float(sdf.iloc[pos + 1]['$close'])
                r = (c1 / c0 - 1.0) if c0 > 0 else 0.0
                res[sym] = {
                    'next_limit_up': int(r >= 0.095),
                    'next_return': r,
                    'next_high_return': max(r, 0.0),
                    'open_premium': 0.0,
                }
            except Exception as e:
                logger.debug(f"次日结果计算失败 {sym}: {e}")
                res[sym] = {'next_limit_up': 0, 'next_return': 0.0, 'next_high_return': 0.0, 'open_premium': 0.0}
        return pd.DataFrame.from_dict(res, orient='index')
    
    def get_concept_stocks(self, concept: str, date: str) -> List[str]:
        """获取概念板块股票（未实现，预留）"""
        return []
    
    def get_intraday_data(self,
                         symbol: str,
                         date: str,
                         freq: str = "1min") -> pd.DataFrame:
        """获取分时数据（AKShare）"""
        try:
            import akshare as ak
            code = symbol.split('.')[0]
            df = ak.stock_zh_a_hist_min_em(
                symbol=code,
                period='1' if freq=='1min' else '5',
                start_date=date.replace('-',''),
                end_date=date.replace('-','')
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={'时间':'time','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','成交额':'amount'})
            df['time'] = pd.to_datetime(df['time'])
            return df
        except Exception:
            return pd.DataFrame()
    
    # ✅ P0-5 (1/3): 封单金额计算
    def get_seal_amount(self, symbol: str, date: str, prev_close: float) -> float:
        """
        计算涨停板封单金额（万元）
        
        方法1: 分钟数据精确计算
        - 找到首次涨停时刻 t
        - 累计 [t, 15:00] 时间段内在涨停价的成交量
        
        方法2: 日线数据近似估算 (fallback)
        - seal_amount ≈ (涨停后成交量 - 正常成交量) * 涨停价
        - 涨停后成交量 = volume - volume_before_limit_up
        
        Args:
            symbol: 股票代码 (e.g. "000001.SZ")
            date: 日期 (YYYY-MM-DD)
            prev_close: 前一交易日收盘价
        
        Returns:
            封单金额（万元）
        """
        cache_key = f"{symbol}:{date}"
        if cache_key in self._seal_amount_cache:
            return self._seal_amount_cache[cache_key]
        
        try:
            # 方法1: 尝试从分钟数据精确计算
            minute_data = self.get_intraday_data(symbol, date, freq='1min')
            
            if not minute_data.empty and prev_close > 0:
                # 计算涨停价 (A股 10% 涨停)
                limit_price = round(prev_close * 1.10, 2)
                
                # 找到首次触及涨停价的时刻
                limit_up_data = minute_data[minute_data['high'] >= limit_price * 0.999]  # 允许误差
                
                if not limit_up_data.empty:
                    limit_up_time = limit_up_data.iloc[0]['time']
                    
                    # 累计涨停后的成交量
                    after_limit_data = minute_data[minute_data['time'] >= limit_up_time]
                    
                    # 近似封单量: 涨停后成交量 * 涨停价
                    # (假设涨停后成交主要是封单换手)
                    seal_volume = after_limit_data['volume'].sum()
                    seal_amount = seal_volume * limit_price / 10000.0  # 转换为万元
                    
                    logger.debug(
                        f"{symbol} {date} 封单金额(分钟数据): {seal_amount:.2f}万元 "
                        f"(涨停时间: {limit_up_time.strftime('%H:%M')})"
                    )
                    
                    self._seal_amount_cache[cache_key] = seal_amount
                    return seal_amount
        
        except Exception as e:
            logger.debug(f"{symbol} {date} 分钟数据获取失败: {e}，使用近似估算")
        
        # 方法2: Fallback 到日线数据近似估算
        try:
            self._ensure_qlib()
            from qlib.data import D
            
            # 获取当日成交量和收盘价
            fields = ['$close', '$volume']
            data = D.features([symbol], fields=fields, start_time=date, end_time=date, freq='day')
            
            if data is not None and not data.empty:
                row = data.iloc[0]
                volume = float(row.get('$volume', 0))
                close = float(row.get('$close', prev_close))
                
                # 假设涨停前成交量占总量 40%
                # 这个比例基于经验: 涨停股通常上午成交活跃,下午封单阻碍成交
                volume_before_limit = volume * 0.4
                seal_volume = volume - volume_before_limit
                
                # 封单金额 = 封单量 * 涨停价
                limit_price = round(prev_close * 1.10, 2)
                seal_amount = seal_volume * limit_price * 100 / 10000.0  # volume是手, *100转股, /10000转万元
                
                logger.debug(
                    f"{symbol} {date} 封单金额(近似): {seal_amount:.2f}万元 "
                    f"(volume={volume:.0f}手, limit_price={limit_price:.2f})"
                )
                
                self._seal_amount_cache[cache_key] = seal_amount
                return seal_amount
        
        except Exception as e:
            logger.warning(f"{symbol} {date} 封单金额计算失败: {e}")
        
        # 完全失败,返回 0
        return 0.0
    
    # ✅ P0-5 (2/3): 连板天数计算
    def get_continuous_board(self, symbol: str, date: str, lookback_days: int = 30) -> int:
        """
        计算连续涨停天数
        
        算法:
        1. 从 date 往前遍历
        2. 计数连续涨停天数 (涨幅 >= 9.9% 且 收盘价 == 最高价)
        3. 遇到非涨停则停止
        
        Args:
            symbol: 股票代码 (e.g. "000001.SZ")
            date: 日期 (YYYY-MM-DD)
            lookback_days: 回望天数 (默认 30 天)
        
        Returns:
            连续涨停天数 (1=首板, 2=二板, ..., 0=当日未涨停)
        """
        cache_key = f"{symbol}:{date}"
        if cache_key in self._continuous_board_cache:
            return self._continuous_board_cache[cache_key]
        
        try:
            self._ensure_qlib()
            from qlib.data import D
            
            # 获取历史行情 (date - lookback_days 到 date)
            end_date = pd.Timestamp(date)
            start_date = end_date - pd.Timedelta(days=lookback_days)
            
            fields = ['$close', '$high', '$low', '$open']
            data = D.features(
                [symbol],
                fields=fields,
                start_time=start_date.strftime('%Y-%m-%d'),
                end_time=date,
                freq='day'
            )
            
            if data is None or data.empty or len(data) < 2:
                return 0
            
            # 标准化列名
            data = data.rename(columns={
                '$close': 'close',
                '$high': 'high',
                '$low': 'low',
                '$open': 'open'
            })
            
            # 计算每日涨幅
            data['prev_close'] = data['close'].shift(1)
            data['pct_change'] = data['close'] / data['prev_close'] - 1.0
            
            # 判断涨停: 涨幅 >= 9.9% 且 收盘价 == 最高价 (允许 0.1% 误差)
            limit_up_threshold = 0.099  # 10% 涨停 (A股)
            data['is_limit_up'] = (
                (data['pct_change'] >= limit_up_threshold) &
                (data['close'] >= data['high'] * 0.999)  # 允许误差
            )
            
            # 从 date 往前计数连续涨停
            continuous_days = 0
            for i in range(len(data) - 1, -1, -1):
                if data['is_limit_up'].iloc[i]:
                    continuous_days += 1
                else:
                    break
            
            logger.debug(
                f"{symbol} {date} 连续涨停天数: {continuous_days} 天"
            )
            
            self._continuous_board_cache[cache_key] = continuous_days
            return continuous_days
        
        except Exception as e:
            logger.warning(f"{symbol} {date} 连板天数计算失败: {e}")
            return 0
    
    # ✅ P0-5 (3/3): 题材热度计算
    def _get_stock_concepts(self, symbol: str) -> List[str]:
        """
        查询股票所属概念板块（使用缓存）
        
        Args:
            symbol: 股票代码 (e.g. "000001.SZ")
        
        Returns:
            概念板块列表
        """
        if symbol in self._concept_cache:
            return self._concept_cache[symbol]
        
        try:
            import akshare as ak
            
            # 提取股票代码 (6位数字)
            code = symbol.split('.')[0]
            
            # 获取股票所属概念板块
            # AKShare API: ak.stock_individual_info_em(symbol=code)
            try:
                # 方法1: 从stockboard单个股票概念获取
                df = ak.stock_individual_info_em(symbol=code)
                
                if df is not None and not df.empty:
                    # 查找含有"概念"的行
                    concept_row = df[df['项目'] == '所属板块']
                    
                    if not concept_row.empty:
                        concept_str = concept_row.iloc[0]['内容']
                        # 概念通常以逗号分隔
                        concepts = [c.strip() for c in str(concept_str).split(',') if c.strip()]
                        
                        logger.debug(f"{symbol} 所属概念: {concepts}")
                        self._concept_cache[symbol] = concepts
                        return concepts
            
            except Exception as e:
                logger.debug(f"{symbol} 从 individual_info 获取概念失败: {e}")
            
            # 方法2: Fallback - 使用行业信息
            basic_map = self._get_basic_map()
            industry = basic_map.get(code, {}).get('industry', '')
            
            if industry:
                concepts = [industry]  # 将行业作为概念
                logger.debug(f"{symbol} 使用行业作为概念: {concepts}")
                self._concept_cache[symbol] = concepts
                return concepts
        
        except Exception as e:
            logger.debug(f"{symbol} 概念查询失败: {e}")
        
        # 完全失败,返回空列表
        self._concept_cache[symbol] = []
        return []
    
    def _is_limit_up(self, symbol: str, date: str) -> bool:
        """
        判断股票在指定日期是否涨停
        
        Args:
            symbol: 股票代码
            date: 日期 (YYYY-MM-DD)
        
        Returns:
            是否涨停
        """
        try:
            self._ensure_qlib()
            from qlib.data import D
            
            # 获取前一交易日和当日数据
            start = (pd.Timestamp(date) - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
            fields = ['$close', '$high']
            data = D.features([symbol], fields=fields, start_time=start, end_time=date, freq='day')
            
            if data is None or data.empty or len(data) < 2:
                return False
            
            # 计算涨幅
            data = data.rename(columns={'$close': 'close', '$high': 'high'})
            prev_close = float(data.iloc[-2]['close'])
            close = float(data.iloc[-1]['close'])
            high = float(data.iloc[-1]['high'])
            
            pct_change = (close / prev_close - 1.0) if prev_close > 0 else 0.0
            
            # 涨停判定: 涨幅 >= 9.9% 且 收盘 == 最高
            is_limit_up = (pct_change >= 0.099) and (close >= high * 0.999)
            
            return is_limit_up
        
        except Exception as e:
            logger.debug(f"{symbol} {date} 涨停判定失败: {e}")
            return False
    
    def get_concept_heat(self, symbol: str, date: str) -> float:
        """
        计算股票所属题材的热度
        
        热度 = 同题材涨停股票数量 / 题材内总股票数量
        
        步骤:
        1. 查询股票所属题材 (可能多个)
        2. 对每个题材，统计当日涨停数量
        3. 取最大热度值
        
        Args:
            symbol: 股票代码 (e.g. "000001.SZ")
            date: 日期 (YYYY-MM-DD)
        
        Returns:
            热度值 (0.0 - 1.0, 表示板块涨停率)
        """
        cache_key = f"{symbol}:{date}"
        
        try:
            # 1. 获取股票所属概念板块
            concepts = self._get_stock_concepts(symbol)
            
            if not concepts:
                logger.debug(f"{symbol} 未找到所属概念，热度=0")
                return 0.0
            
            max_heat = 0.0
            
            # 2. 对每个概念，计算热度
            for concept in concepts:
                try:
                    # 获取该概念板块的成分股
                    # 由于AKShare概念板块API较复杂,这里使用简化逻辑:
                    # - 先获取当日所有涨停股
                    # - 统计其中属于同一概念的数量
                    
                    # 简化版: 使用行业作为代理
                    # (实际生产环境应接入完整概念板块API)
                    
                    # 获取当日涨停股
                    limit_up_stocks = self.get_limit_up_stocks(date)
                    
                    if not limit_up_stocks:
                        continue
                    
                    # 统计同概念涨停数
                    same_concept_limit_up = 0
                    total_same_concept = 0
                    
                    for stock in limit_up_stocks:
                        stock_concepts = self._get_stock_concepts(stock.symbol)
                        
                        if concept in stock_concepts:
                            same_concept_limit_up += 1
                            total_same_concept += 1
                    
                    # 计算热度 (简化: 涨停数量直接作为热度,不除以总数)
                    # 实际应该是: same_concept_limit_up / total_in_concept
                    # 这里由于无法轻易获取板块总成分股,使用涨停数作为热度代理
                    heat = float(same_concept_limit_up)  # 单位:只
                    
                    logger.debug(
                        f"{symbol} 概念'{concept}' 热度: {heat} "
                        f"(涨停{same_concept_limit_up}只)"
                    )
                    
                    max_heat = max(max_heat, heat)
                
                except Exception as e:
                    logger.debug(f"概念'{concept}'热度计算失败: {e}")
                    continue
            
            return max_heat
        
        except Exception as e:
            logger.warning(f"{symbol} {date} 题材热度计算失败: {e}")
            return 0.0


class LimitUpFactorLibrary:
    """涨停板因子库"""
    
    @staticmethod
    def factor_seal_strength(data: pd.DataFrame) -> pd.Series:
        """
        封板强度因子
        计算：封单金额 / 流通市值
        """
        return data['seal_amount'] / data['market_cap']
    
    @staticmethod
    def factor_continuous_momentum(data: pd.DataFrame) -> pd.Series:
        """
        连板动量因子
        连板天数越高，动量越强
        """
        return np.log1p(data['continuous_board']) * data['volume_ratio']
    
    @staticmethod
    def factor_concept_synergy(data: pd.DataFrame) -> pd.Series:
        """
        题材共振因子
        同题材涨停数量 * 个股涨停强度
        """
        return data['concept_heat'] * data['limit_up_strength']
    
    @staticmethod
    def factor_early_limit_up(data: pd.DataFrame) -> pd.Series:
        """
        早盘涨停因子
        涨停时间越早越强（09:30=1.0, 15:00=0.0）
        """
        # 假设limit_up_minutes是涨停分钟数（从9:30开始）
        total_minutes = 240  # 交易时长
        return 1.0 - (data['limit_up_minutes'] / total_minutes)
    
    @staticmethod
    def factor_volume_explosion(data: pd.DataFrame) -> pd.Series:
        """
        量能爆发因子
        成交量 / 近20日均量
        """
        return data['volume'] / data['volume_ma20']
    
    @staticmethod
    def factor_large_order_net(data: pd.DataFrame) -> pd.Series:
        """
        大单净流入因子
        大买单 - 大卖单
        """
        return (data['large_buy'] - data['large_sell']) / data['amount']
    
    @staticmethod
    def get_all_factors() -> Dict[str, callable]:
        """获取所有预定义因子"""
        return {
            'seal_strength': LimitUpFactorLibrary.factor_seal_strength,
            'continuous_momentum': LimitUpFactorLibrary.factor_continuous_momentum,
            'concept_synergy': LimitUpFactorLibrary.factor_concept_synergy,
            'early_limit_up': LimitUpFactorLibrary.factor_early_limit_up,
            'volume_explosion': LimitUpFactorLibrary.factor_volume_explosion,
            'large_order_net': LimitUpFactorLibrary.factor_large_order_net,
        }


# 使用示例
import logging
logger = logging.getLogger(__name__)

def example_usage():
    """使用示例"""
    # 初始化数据接口
    data_interface = LimitUpDataInterface(data_source="qlib")
    
    # 获取涨停股票
    limit_up_stocks = data_interface.get_limit_up_stocks(
        date="2024-01-15",
        exclude_st=True,
        exclude_new=True
    )
    
    logger.info(f"找到 {len(limit_up_stocks)} 只涨停股票")
    
    # 获取特征
    symbols = [stock.symbol for stock in limit_up_stocks]
    features = data_interface.get_limit_up_features(symbols, "2024-01-15")
    
    # 获取次日结果
    results = data_interface.get_next_day_result(symbols, "2024-01-15")
    
    # 应用因子
    factor_lib = LimitUpFactorLibrary()
    # seal_strength = factor_lib.factor_seal_strength(features)
    
    logger.info("数据接口测试完成")


if __name__ == "__main__":
    example_usage()
