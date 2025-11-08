"""Qlib DataHandler集成CZSC缠论特征"""

from qlib.data.dataset.handler import DataHandlerLP
from features.chanlun.czsc_features import CzscFeatureGenerator
import pandas as pd
import logging
import hashlib
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class CzscChanLunHandler(DataHandlerLP):
    """
    CZSC缠论特征Handler
    
    功能:
    - 集成CZSC缠论特征到Qlib
    - 支持批量股票处理
    - 自动缓存结果
    
    使用方法:
        在Qlib配置文件中:
        handler:
            class: CzscChanLunHandler
            module_path: qlib_enhanced.chanlun.czsc_handler
            kwargs:
                start_time: "2020-01-01"
                end_time: "2023-12-31"
                instruments: "csi300"
                freq: "day"
                drop_raw: false  # 是否删除原始OHLCV数据
    """
    
    def __init__(self, 
                 instruments='csi300', 
                 start_time=None, 
                 end_time=None,
                 freq='day', 
                 infer_processors=[], 
                 learn_processors=[],
                 fit_start_time=None, 
                 fit_end_time=None, 
                 process_type=DataHandlerLP.PTYPE_A,
                 drop_raw=False,
                 enable_cache=False,
                 cache_config: Optional[Dict] = None,
                 enable_parallel=False,
                 parallel_config: Optional[Dict] = None,
                 **kwargs):
        
        self.freq = freq
        self.drop_raw = drop_raw
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel
        
        # 初始化CZSC特征生成器
        czsc_freq_map = {
            'day': '日线',
            '60min': '60分',
            '30min': '30分',
            '15min': '15分',
        }
        self.czsc_gen = CzscFeatureGenerator(freq=czsc_freq_map.get(freq, '日线'))
        
        # 初始化缓存
        self.cache = None
        if enable_cache and cache_config:
            try:
                from .chanlun_cache import create_cache_from_config
                self.cache = create_cache_from_config(cache_config)
                logger.info("✅ 缓存已启用")
            except Exception as e:
                logger.warning(f"缓存初始化失败: {e}, 将禁用缓存")
                self.enable_cache = False
        
        # 初始化并行处理器
        self.batch_processor = None
        if enable_parallel and parallel_config:
            try:
                from .chanlun_parallel import ChanLunBatchProcessor
                self.batch_processor = ChanLunBatchProcessor(
                    cache=self.cache,
                    enable_cache=enable_cache,
                    enable_parallel=True,
                    **parallel_config
                )
                logger.info("✅ 并行处理已启用")
            except Exception as e:
                logger.warning(f"并行处理器初始化失败: {e}, 将禁用并行")
                self.enable_parallel = False
        
        # 定义需要加载的基础字段
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self._get_base_fields(),
                "freq": freq,
            },
        }
        
        # 默认处理器
        if not infer_processors:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
        
        if not learn_processors:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ]
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            process_type=process_type,
            **kwargs
        )
    
    def _get_base_fields(self):
        """定义基础字段"""
        fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
        names = ["open", "close", "high", "low", "volume", "factor"]
        
        # 添加标签: 未来1日收益率
        fields.append("Ref($close, -2)/Ref($close, -1) - 1")
        names.append("LABEL0")
        
        return fields, names
    
    def fetch(self, selector=None, level=None, col_set=None):
        """重写fetch方法, 添加CZSC缠论特征"""
        # 1. 获取基础OHLCV数据
        df = super().fetch(selector=selector, level=level, col_set=col_set)
        
        if df is None or len(df) == 0:
            logger.warning("基础数据为空")
            return df
        
        logger.info(f"开始计算CZSC缠论特征, 股票数: {len(df.index.get_level_values(0).unique())}")
        
        # 2. 准备任务
        instruments = df.index.get_level_values(0).unique()
        tasks = []
        for instrument in instruments:
            inst_df = df.loc[instrument].reset_index()
            tasks.append({
                'symbol': instrument,
                'data': inst_df
            })
        
        # 3. 处理任务 (支持缓存+并行)
        if self.batch_processor and len(tasks) > 1:
            # 使用批量处理器 (集成缓存+并行)
            czsc_features_list = self.batch_processor.process_batch(
                tasks=tasks,
                process_func=self._process_single_stock,
                cache_key_func=lambda t: self._make_cache_key(t['symbol'], t['data']),
                progress_callback=None
            )
            # 过滤None结果
            czsc_features_list = [r for r in czsc_features_list if r is not None]
        else:
            # 串行处理 (可能使用缓存)
            czsc_features_list = []
            for task in tasks:
                try:
                    # 检查缓存
                    cache_key = self._make_cache_key(task['symbol'], task['data'])
                    if self.enable_cache and self.cache:
                        cached = self.cache.get(cache_key)
                        if cached is not None:
                            czsc_features_list.append(cached)
                            continue
                    
                    # 计算特征
                    result = self._process_single_stock(task)
                    if result is not None:
                        czsc_features_list.append(result)
                        
                        # 写入缓存
                        if self.enable_cache and self.cache:
                            self.cache.set(cache_key, result)
                    
                except Exception as e:
                    logger.error(f"股票{task['symbol']}缠论特征计算失败: {e}")
                    continue
        
        if not czsc_features_list:
            logger.warning("无缠论特征生成")
            return df
        
        # 3. 合并缠论特征
        czsc_df = pd.concat(czsc_features_list, ignore_index=True)
        czsc_df = czsc_df.set_index(['instrument', 'datetime'])
        
        # 4. 添加缠论特征列到原始DataFrame
        feature_cols = ['fx_mark', 'bi_direction', 'bi_position', 
                       'bi_power', 'in_zs', 'bars_since_fx']
        
        for col in feature_cols:
            if col in czsc_df.columns:
                df[col] = czsc_df[col]
        
        # 5. 可选: 删除原始OHLCV (节省存储)
        if self.drop_raw:
            df = df.drop(columns=['open', 'high', 'low'], errors='ignore')
        
        logger.info(f"✅ CZSC缠论特征计算完成, 新增特征: {len(feature_cols)}")
        
        # 6. 输出缓存统计
        if self.enable_cache and self.cache:
            stats = self.cache.get_stats()
            logger.info(
                f"缓存统计: 命中率={stats.hit_rate:.1%}, "
                f"命中={stats.hits}, 未命中={stats.misses}"
            )
        
        return df
    
    def _process_single_stock(self, task: Dict) -> Optional[pd.DataFrame]:
        """处理单个股票 (用于并行)"""
        try:
            inst_df = task['data']
            instrument = task['symbol']
            
            # 准备CZSC输入格式
            czsc_input = pd.DataFrame({
                'datetime': inst_df['datetime'],
                'open': inst_df['open'],
                'close': inst_df['close'],
                'high': inst_df['high'],
                'low': inst_df['low'],
                'volume': inst_df['volume'],
                'symbol': instrument
            })
            
            # 生成缠论特征
            czsc_result = self.czsc_gen.generate_features(czsc_input)
            czsc_result['instrument'] = instrument
            czsc_result['datetime'] = inst_df['datetime'].values
            
            return czsc_result
        except Exception as e:
            logger.error(f"股票{task['symbol']}处理失败: {e}")
            return None
    
    def _make_cache_key(self, symbol: str, data: pd.DataFrame) -> str:
        """生成缓存键 (基于symbol和数据hash)"""
        # 使用symbol + 数据行数 + 首尾日期 作为键
        if len(data) == 0:
            return f"{symbol}_empty"
        
        first_date = str(data['datetime'].iloc[0]) if 'datetime' in data.columns else 'unknown'
        last_date = str(data['datetime'].iloc[-1]) if 'datetime' in data.columns else 'unknown'
        
        return f"{symbol}_{len(data)}_{first_date}_{last_date}"
