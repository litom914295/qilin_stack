"""Chan.py买卖点特征提取器"""

import sys
from pathlib import Path
# 添加chanpy到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'chanpy'))
# 添加qlib_enhanced模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE, BSP_TYPE
import pandas as pd
import logging
import os
from qlib_enhanced.chanlun.trend_classifier import TrendClassifier
# P1-1: 中枢扩展/升级/移动分析器
try:
    from ZS.ZSAnalyzer import ZSAnalyzer
except Exception:
    ZSAnalyzer = None
# P2-1: 多周期共振评分
try:
    from qlib_enhanced.chanlun.multi_timeframe_confluence import (
        resample_ohlc, compute_direction, compute_confluence_score,
    )
except Exception:
    resample_ohlc = None
    compute_direction = None
    compute_confluence_score = None

logger = logging.getLogger(__name__)

class ChanPyFeatureGenerator:
    """
    Chan.py缠论特征生成器
    
    功能:
    - 买卖点识别 (6类)
    - 线段识别
    - 完整中枢识别
    - 背驰判断
    """
    
    def __init__(self, seg_algo='chan', bi_algo='normal', zs_combine=True):
        """
        Args:
            seg_algo: 线段算法 ('chan'/'def'/'dyh')
            bi_algo: 笔算法 ('normal'/'new'/'amplitude')
            zs_combine: 是否合并中枢
        """
        self.config = CChanConfig({
            'seg_algo': seg_algo,
            'bi_algo': bi_algo,
            'zs_combine': zs_combine,
            'trigger_step': False,
        })
        # 确保临时目录存在
        os.makedirs('G:/test/qilin_stack/temp', exist_ok=True)
        # P0-1: 走势类型识别器
        self.trend_classifier = TrendClassifier()
    
    def generate_features(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        生成Chan.py缠论特征
        
        Args:
            df: DataFrame with [datetime, open, close, high, low, volume]
            code: 股票代码
        
        Returns:
            df with Chan.py特征
        """
        if len(df) < 20:
            logger.warning(f"{code}: 数据不足20条, 跳过Chan.py计算")
            return self._add_empty_features(df)
        
        try:
            # 1. 保存临时CSV (使用Windows路径)
            temp_csv = f'G:/test/qilin_stack/temp/chanpy_{code.replace(".", "_")}.csv'
            df[['datetime', 'open', 'close', 'high', 'low', 'volume']].to_csv(
                temp_csv, index=False
            )
            
            # 2. 创建CChan实例
            chan = CChan(
                code=code,
                begin_time=str(df['datetime'].iloc[0]),
                end_time=str(df['datetime'].iloc[-1]),
                data_src='csvAPI',  # 使用我们的CSV适配器
                lv_list=[KL_TYPE.K_DAY],
                config=self.config
            )
            
            # 3. 提取特征
            result = df.copy()
            
            # 买卖点特征
            bsp_features = self._extract_bsp_features(chan[0], df)
            result = pd.merge(result, bsp_features, on='datetime', how='left')
            
            # 线段特征
            seg_features = self._extract_seg_features(chan[0], df)
            result = pd.merge(result, seg_features, on='datetime', how='left')
            
            # 中枢特征
            zs_features = self._extract_zs_features(chan[0], df)
            result = pd.merge(result, zs_features, on='datetime', how='left')
            
            # P0-1: 走势类型特征
            trend_features = self._extract_trend_features(chan[0], df)
            result = pd.merge(result, trend_features, on='datetime', how='left')

            # P1-1: 中枢扩展/升级/移动高级特征
            zs_adv = self._extract_zs_advanced_features(chan[0], df)
            result = pd.merge(result, zs_adv, on='datetime', how='left')

            # P2-1: 多周期共振分数
            conflu = self._extract_confluence_features(df)
            result = pd.merge(result, conflu, on='datetime', how='left')
            
            # 清理临时文件
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            
            return result
            
        except Exception as e:
            logger.error(f"{code}: Chan.py特征生成失败: {e}", exc_info=True)
            return self._add_empty_features(df)
    
    def _extract_bsp_features(self, kl_list, df) -> pd.DataFrame:
        """提取买卖点特征"""
        try:
            bsp_list = kl_list.bs_point_lst.lst if hasattr(kl_list, 'bs_point_lst') else []
        except:
            bsp_list = []
        
        features = []
        for _, row in df.iterrows():
            feat = {
                'datetime': row['datetime'],
                'is_buy_point': 0,
                'is_sell_point': 0,
                'bsp_type': 0,  # 0=无, 1/2/3=类型
                'bsp_is_buy': 0,  # 1=买, -1=卖
            }
            
            # 查找对应日期的买卖点
            for bsp in bsp_list:
                try:
                    bsp_time = pd.to_datetime(str(bsp.klu.time))
                    row_time = pd.to_datetime(row['datetime'])
                    
                    if bsp_time.date() == row_time.date():
                        feat['is_buy_point'] = 1 if bsp.is_buy else 0
                        feat['is_sell_point'] = 0 if bsp.is_buy else 1
                        # 买卖点类型: 1买/1卖, 2买/2卖, 3买/3卖等
                        feat['bsp_type'] = bsp.type2idx() if hasattr(bsp, 'type2idx') else 0
                        feat['bsp_is_buy'] = 1 if bsp.is_buy else -1
                        break
                except Exception as e:
                    continue
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _extract_seg_features(self, kl_list, df) -> pd.DataFrame:
        """提取线段特征"""
        try:
            seg_list = kl_list.seg_list if hasattr(kl_list, 'seg_list') else []
        except:
            seg_list = []
        
        features = []
        for _, row in df.iterrows():
            feat = {
                'datetime': row['datetime'],
                'seg_direction': 0,  # 1=上, -1=下
                'is_seg_start': 0,
                'is_seg_end': 0,
            }
            
            # 查找所在线段
            for seg in seg_list:
                try:
                    seg_start = pd.to_datetime(str(seg.start_bi.get_begin_klu().time))
                    seg_end = pd.to_datetime(str(seg.end_bi.get_end_klu().time))
                    row_date = pd.to_datetime(row['datetime'])
                    
                    if seg_start.date() <= row_date.date() <= seg_end.date():
                        feat['seg_direction'] = 1 if seg.is_up() else -1
                        feat['is_seg_start'] = 1 if row_date.date() == seg_start.date() else 0
                        feat['is_seg_end'] = 1 if row_date.date() == seg_end.date() else 0
                        break
                except Exception as e:
                    continue
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _extract_zs_features(self, kl_list, df) -> pd.DataFrame:
        """提取中枢特征"""
        try:
            seg_list = kl_list.seg_list if hasattr(kl_list, 'seg_list') else []
        except:
            seg_list = []
        
        features = []
        for _, row in df.iterrows():
            feat = {
                'datetime': row['datetime'],
                'in_chanpy_zs': 0,  # 与CZSC区分
                'zs_low_chanpy': None,
                'zs_high_chanpy': None,
            }
            
            # 查找中枢
            for seg in seg_list:
                try:
                    zs_list = seg.zs_lst if hasattr(seg, 'zs_lst') else []
                    for zs in zs_list:
                        zs_start = pd.to_datetime(str(zs.begin.time))
                        zs_end = pd.to_datetime(str(zs.end.time))
                        row_date = pd.to_datetime(row['datetime'])
                        
                        if zs_start.date() <= row_date.date() <= zs_end.date():
                            feat['in_chanpy_zs'] = 1
                            feat['zs_low_chanpy'] = zs.low
                            feat['zs_high_chanpy'] = zs.high
                            break
                except Exception as e:
                    continue
            
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _extract_trend_features(self, kl_list, df) -> pd.DataFrame:
        """P0-1: 提取走势类型特征"""
        try:
            seg_list = kl_list.seg_list if hasattr(kl_list, 'seg_list') else []
        except:
            seg_list = []
        
        # 提取中枢列表
        zs_list = []
        for seg in seg_list:
            try:
                if hasattr(seg, 'zs_lst'):
                    zs_list.extend(seg.zs_lst)
            except:
                pass
        
        # 使用TrendClassifier识别走势类型
        if len(seg_list) >= 3:
            trend_result = self.trend_classifier.classify_with_details(seg_list, zs_list)
            trend_type_value = {'UPTREND': 1, 'DOWNTREND': -1, 'SIDEWAYS': 0, 'UNKNOWN': 0}.get(trend_result['trend_type'], 0)
            trend_strength = trend_result['strength']
        else:
            trend_type_value = 0
            trend_strength = 0.0
        
        # 为每行添加特征
        features = []
        for _, row in df.iterrows():
            feat = {
                'datetime': row['datetime'],
                'trend_type': trend_type_value,       # P0-1: 走势类型
                'trend_strength': trend_strength,     # P0-1: 走势强度
            }
            features.append(feat)
        
        return pd.DataFrame(features)
    
    def _extract_zs_advanced_features(self, kl_list, df) -> pd.DataFrame:
        """P1-1: 提取中枢扩展/升级/移动高级特征
        输出(按行填充同一值，表示当前窗口状态):
        - zs_movement_direction: -1/0/1 (下/横/上)
        - zs_movement_slope: float
        - zs_movement_confidence: float [0,1]
        - zs_upgrade_flag: 0/1
        - zs_upgrade_strength: float [0,1]
        """
        # 收集线段与中枢
        try:
            seg_list = kl_list.seg_list if hasattr(kl_list, 'seg_list') else []
        except Exception:
            seg_list = []
        zs_list = []
        for seg in seg_list:
            try:
                if hasattr(seg, 'zs_lst') and seg.zs_lst:
                    zs_list.extend(seg.zs_lst)
            except Exception:
                pass
        # 默认值
        direction_v = 0
        slope_v = 0.0
        conf_v = 0.0
        upg_flag = 0
        upg_strength = 0.0
        # 计算移动
        if ZSAnalyzer is not None and len(zs_list) >= 3:
            try:
                analyzer = ZSAnalyzer()
                move = analyzer.analyze_zs_movement(zs_list)
                direction_v = {'rising':1, 'falling':-1}.get(move.get('direction','sideways'), 0)
                slope_v = float(move.get('slope', 0) or 0)
                conf_v = float(move.get('confidence', 0) or 0)
                # 升级检测
                upg = analyzer.detect_zs_upgrade(seg_list)
                if upg is not None:
                    upg_flag = 1
                    upg_strength = float(getattr(upg, 'strength', 0) or 0)
            except Exception:
                pass
        # 展开到每行
        feats = []
        for _, row in df.iterrows():
            feats.append({
                'datetime': row['datetime'],
                'zs_movement_direction': direction_v,
                'zs_movement_slope': slope_v,
                'zs_movement_confidence': conf_v,
                'zs_upgrade_flag': upg_flag,
                'zs_upgrade_strength': upg_strength,
            })
        return pd.DataFrame(feats)

    def _extract_confluence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """P2-1: 计算多周期共振分数并逐行填充
        - confluence_score: float (D/W/M 方向按权重求和)
        """
        score = 0.0
        try:
            if compute_direction is not None:
                # 日线方向
                d_dir = compute_direction(df)
                # 周、月重采样
                if resample_ohlc is not None:
                    w_df = resample_ohlc(df, 'W')
                    m_df = resample_ohlc(df, 'M')
                    w_dir = compute_direction(w_df)
                    m_dir = compute_direction(m_df)
                else:
                    w_dir = 0
                    m_dir = 0
                dirs = {'D': d_dir, 'W': w_dir, 'M': m_dir}
                score = float(compute_confluence_score(dirs)) if compute_confluence_score else 0.0
        except Exception:
            score = 0.0
        feats = []
        for _, row in df.iterrows():
            feats.append({'datetime': row['datetime'], 'confluence_score': score})
        return pd.DataFrame(feats)

    def _add_empty_features(self, df) -> pd.DataFrame:
        """添加空特征"""
        result = df.copy()
        empty_features = {
            'is_buy_point': 0,
            'is_sell_point': 0,
            'bsp_type': 0,
            'bsp_is_buy': 0,
            'seg_direction': 0,
            'is_seg_start': 0,
            'is_seg_end': 0,
            'in_chanpy_zs': 0,
            'zs_low_chanpy': None,
            'zs_high_chanpy': None,
            'trend_type': 0,          # P0-1
'trend_strength': 0.0,    # P0-1
            # P1-1
            'zs_movement_direction': 0,
            'zs_movement_slope': 0.0,
            'zs_movement_confidence': 0.0,
            'zs_upgrade_flag': 0,
            'zs_upgrade_strength': 0.0,
            # P2-1
            'confluence_score': 0.0,
        }
        for col, val in empty_features.items():
            result[col] = val
        return result
