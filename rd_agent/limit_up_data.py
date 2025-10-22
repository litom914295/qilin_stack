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
                # 连板天数（向后回溯，连续 ret>=9.5%）
                cont = 0
                for r in reversed(list(sdf['ret'].iloc[-10:])):
                    if r >= 0.095:
                        cont += 1
                    else:
                        break
                # 封板质量 proxy：收盘接近最高 且 K 线下影小
                high = row.get('high', np.nan)
                low = row.get('low', np.nan)
                seal_quality = 0.0
                if np.isfinite(high) and np.isfinite(close) and np.isfinite(low) and high > 0:
                    near_high = 1.0 - (high - close) / max(1e-6, high)
                    lower_shadow = (close - low) / max(1e-6, close)
                seal_quality = float(max(0.0, min(10.0, (near_high * 6.0 + (1.0 - lower_shadow) * 4.0) * 1.0)))
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
                feats[sym] = {
                    'limit_up_strength': float(strength),
                    'seal_quality': float(seal_quality),
'concept_heat': 0.0,  # 后续用行业共振估计
                    'continuous_board': int(cont),
'volume_surge': float(volume_surge),
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
        # 行业共振（作为“题材热度”）
        try:
            limitups_today = self.get_limit_up_stocks(date)
            ind_map = {rec.symbol.split('.')[0]: basic_map.get(rec.symbol.split('.')[0], {}).get('industry', '') for rec in limitups_today}
            # 统计每个行业今天涨停数量
            from collections import Counter
            ind_counter = Counter(ind_map.values())
            for sym in feats.keys():
                ind = feats[sym].get('industry', '')
                feats[sym]['concept_heat'] = float(ind_counter.get(ind, 0))
        except Exception:
            pass
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
    
    print(f"找到 {len(limit_up_stocks)} 只涨停股票")
    
    # 获取特征
    symbols = [stock.symbol for stock in limit_up_stocks]
    features = data_interface.get_limit_up_features(symbols, "2024-01-15")
    
    # 获取次日结果
    results = data_interface.get_next_day_result(symbols, "2024-01-15")
    
    # 应用因子
    factor_lib = LimitUpFactorLibrary()
    # seal_strength = factor_lib.factor_seal_strength(features)
    
    print("数据接口测试完成")


if __name__ == "__main__":
    example_usage()
