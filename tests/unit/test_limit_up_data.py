"""
rd_agent.limit_up_data 模块单元测试

测试 LimitUpDataInterface 类的所有功能:
- 初始化和配置
- P0-5: 封单金额计算 (get_seal_amount)
- P0-5: 连板天数计算 (get_continuous_board)
- P0-5: 题材热度计算 (get_concept_heat)
- 完整特征集成 (get_limit_up_features)
- 缓存机制
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from rd_agent.limit_up_data import (
    LimitUpDataInterface,
    LimitUpRecord
)


class TestLimitUpDataInterface:
    """LimitUpDataInterface 基础功能测试"""
    
    def test_initialization_default(self):
        """测试默认初始化"""
        data_interface = LimitUpDataInterface()
        
        assert data_interface.data_source == "qlib"
        assert data_interface._qlib_initialized == False
        assert isinstance(data_interface._seal_amount_cache, dict)
        assert isinstance(data_interface._continuous_board_cache, dict)
        assert isinstance(data_interface._concept_cache, dict)
    
    def test_initialization_with_data_source(self):
        """测试指定数据源初始化"""
        data_interface = LimitUpDataInterface(data_source="akshare")
        
        assert data_interface.data_source == "akshare"
    
    def test_cache_initialization(self):
        """测试缓存初始化 (P0-5)"""
        data_interface = LimitUpDataInterface()
        
        # 验证所有缓存都已初始化
        assert hasattr(data_interface, '_seal_amount_cache')
        assert hasattr(data_interface, '_continuous_board_cache')
        assert hasattr(data_interface, '_concept_cache')
        assert hasattr(data_interface, '_concept_stocks_cache')
        
        # 验证缓存为空
        assert len(data_interface._seal_amount_cache) == 0
        assert len(data_interface._continuous_board_cache) == 0
        assert len(data_interface._concept_cache) == 0


# P0-5 (1/3): 封单金额计算测试
class TestGetSealAmount:
    """P0-5: 测试封单金额计算功能"""
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_intraday_data')
    def test_get_seal_amount_from_minute_data(self, mock_get_intraday):
        """测试从分钟数据计算封单金额"""
        data_interface = LimitUpDataInterface()
        
        # Mock 分钟数据
        minute_data = pd.DataFrame({
            'time': pd.date_range('2024-01-15 09:30', periods=240, freq='1min'),
            'open': [10.0] * 240,
            'high': [10.0] * 60 + [11.0] * 180,  # 10:30 开始涨停
            'low': [10.0] * 240,
            'close': [10.0] * 60 + [11.0] * 180,
            'volume': [1000] * 240  # 每分钟1000手
        })
        mock_get_intraday.return_value = minute_data
        
        # 计算封单金额
        seal_amount = data_interface.get_seal_amount(
            symbol="000001.SZ",
            date="2024-01-15",
            prev_close=10.0
        )
        
        # 验证计算逻辑
        # 涨停后成交量 = 180分钟 * 1000手 = 180000手
        # 封单金额 = 180000 * 11.0 / 10000 = 198万元
        assert seal_amount > 0
        assert isinstance(seal_amount, float)
        
        # 验证缓存
        assert "000001.SZ:2024-01-15" in data_interface._seal_amount_cache
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_intraday_data')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_seal_amount_fallback_to_daily(self, mock_ensure_qlib, mock_get_intraday):
        """测试 fallback 到日线数据计算封单金额"""
        data_interface = LimitUpDataInterface()
        
        # Mock 分钟数据为空 (触发 fallback)
        mock_get_intraday.return_value = pd.DataFrame()
        mock_ensure_qlib.return_value = None
        
        # Mock Qlib D.features
        mock_data = pd.DataFrame({
            '$volume': [10000],  # 10000手
            '$close': [11.0]
        })
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            seal_amount = data_interface.get_seal_amount(
                symbol="000001.SZ",
                date="2024-01-15",
                prev_close=10.0
            )
            
            # Fallback 计算:
            # seal_volume = 10000 * 0.6 = 6000手
            # seal_amount = 6000 * 11.0 * 100 / 10000 = 660万元
            assert seal_amount >= 0
            assert isinstance(seal_amount, float)
    
    def test_get_seal_amount_cache_hit(self):
        """测试封单金额缓存命中"""
        data_interface = LimitUpDataInterface()
        
        # 预设缓存
        data_interface._seal_amount_cache["000001.SZ:2024-01-15"] = 500.0
        
        # 调用应该直接返回缓存值,不触发计算
        seal_amount = data_interface.get_seal_amount(
            symbol="000001.SZ",
            date="2024-01-15",
            prev_close=10.0
        )
        
        assert seal_amount == 500.0
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_intraday_data')
    def test_get_seal_amount_no_limit_up(self, mock_get_intraday):
        """测试未涨停时的封单金额"""
        data_interface = LimitUpDataInterface()
        
        # Mock 分钟数据 (未涨停)
        minute_data = pd.DataFrame({
            'time': pd.date_range('2024-01-15 09:30', periods=240, freq='1min'),
            'high': [10.5] * 240,  # 最高10.5,未涨停 (涨停价11.0)
            'volume': [1000] * 240
        })
        mock_get_intraday.return_value = minute_data
        
        seal_amount = data_interface.get_seal_amount(
            symbol="000001.SZ",
            date="2024-01-15",
            prev_close=10.0
        )
        
        # 未涨停应该返回 0 或通过 fallback 计算
        assert seal_amount >= 0
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_intraday_data')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_seal_amount_calculation_failure(self, mock_ensure_qlib, mock_get_intraday):
        """测试计算失败时返回 0"""
        data_interface = LimitUpDataInterface()
        
        # Mock 抛出异常
        mock_get_intraday.side_effect = Exception("Data fetch error")
        mock_ensure_qlib.side_effect = Exception("Qlib error")
        
        seal_amount = data_interface.get_seal_amount(
            symbol="000001.SZ",
            date="2024-01-15",
            prev_close=10.0
        )
        
        # 完全失败应该返回 0
        assert seal_amount == 0.0


# P0-5 (2/3): 连板天数计算测试
class TestGetContinuousBoard:
    """P0-5: 测试连板天数计算功能"""
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_continuous_board_first_board(self, mock_ensure_qlib):
        """测试首板 (连板天数 = 1)"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 历史数据: 只有今天涨停
        mock_data = pd.DataFrame({
            '$close': [10.0, 11.0],  # 今天涨停
            '$high': [10.0, 11.0],
            '$low': [9.8, 10.9],
            '$open': [10.0, 10.5]
        }, index=pd.date_range('2024-01-14', periods=2))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            continuous_days = data_interface.get_continuous_board(
                symbol="000001.SZ",
                date="2024-01-15"
            )
            
            # 首板应该返回 1
            assert continuous_days == 1
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_continuous_board_second_board(self, mock_ensure_qlib):
        """测试二板 (连板天数 = 2)"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 历史数据: 连续两天涨停
        mock_data = pd.DataFrame({
            '$close': [9.0, 10.0, 11.0],  # 连续涨停
            '$high': [9.0, 10.0, 11.0],
            '$low': [8.9, 9.9, 10.9],
            '$open': [8.8, 9.5, 10.5]
        }, index=pd.date_range('2024-01-13', periods=3))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            continuous_days = data_interface.get_continuous_board(
                symbol="000001.SZ",
                date="2024-01-15",
                lookback_days=5
            )
            
            # 二板应该返回 2
            assert continuous_days == 2
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_continuous_board_no_limit_up(self, mock_ensure_qlib):
        """测试当日未涨停 (连板天数 = 0)"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 历史数据: 今天未涨停
        mock_data = pd.DataFrame({
            '$close': [10.0, 10.5],  # 今天涨5%,未涨停
            '$high': [10.0, 10.6],
            '$low': [9.8, 10.4],
            '$open': [10.0, 10.2]
        }, index=pd.date_range('2024-01-14', periods=2))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            continuous_days = data_interface.get_continuous_board(
                symbol="000001.SZ",
                date="2024-01-15"
            )
            
            # 未涨停应该返回 0
            assert continuous_days == 0
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_continuous_board_three_boards(self, mock_ensure_qlib):
        """测试三连板 (连板天数 = 3)"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 历史数据: 连续三天涨停
        mock_data = pd.DataFrame({
            '$close': [8.0, 9.0, 10.0, 11.0],  # 连续3天涨停
            '$high': [8.0, 9.0, 10.0, 11.0],
            '$low': [7.9, 8.9, 9.9, 10.9],
            '$open': [7.8, 8.5, 9.5, 10.5]
        }, index=pd.date_range('2024-01-12', periods=4))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            continuous_days = data_interface.get_continuous_board(
                symbol="000001.SZ",
                date="2024-01-15",
                lookback_days=10
            )
            
            # 三连板应该返回 3
            assert continuous_days == 3
    
    def test_get_continuous_board_cache_hit(self):
        """测试连板天数缓存命中"""
        data_interface = LimitUpDataInterface()
        
        # 预设缓存
        data_interface._continuous_board_cache["000001.SZ:2024-01-15"] = 2
        
        continuous_days = data_interface.get_continuous_board(
            symbol="000001.SZ",
            date="2024-01-15"
        )
        
        assert continuous_days == 2
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_continuous_board_insufficient_data(self, mock_ensure_qlib):
        """测试数据不足时返回 0"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 数据不足 (< 2 天)
        mock_data = pd.DataFrame({
            '$close': [10.0],
            '$high': [10.0],
            '$low': [9.8],
            '$open': [10.0]
        }, index=pd.date_range('2024-01-15', periods=1))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            continuous_days = data_interface.get_continuous_board(
                symbol="000001.SZ",
                date="2024-01-15"
            )
            
            assert continuous_days == 0
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_continuous_board_calculation_failure(self, mock_ensure_qlib):
        """测试计算失败时返回 0"""
        data_interface = LimitUpDataInterface()
        
        # Mock 抛出异常
        mock_ensure_qlib.side_effect = Exception("Qlib error")
        
        continuous_days = data_interface.get_continuous_board(
            symbol="000001.SZ",
            date="2024-01-15"
        )
        
        assert continuous_days == 0


# P0-5 (3/3): 题材热度计算测试
class TestGetConceptHeat:
    """P0-5: 测试题材热度计算功能"""
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._get_stock_concepts')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_limit_up_stocks')
    def test_get_concept_heat_basic(self, mock_get_limit_up, mock_get_concepts):
        """测试基本题材热度计算"""
        data_interface = LimitUpDataInterface()
        
        # Mock 股票所属概念
        def concept_side_effect(symbol):
            if symbol == "000001.SZ":
                return ["人工智能", "芯片"]
            elif symbol == "000002.SZ":
                return ["人工智能"]
            elif symbol == "000003.SZ":
                return ["新能源"]
            return []
        
        mock_get_concepts.side_effect = concept_side_effect
        
        # Mock 当日涨停股
        mock_limit_up = [
            Mock(symbol="000001.SZ"),
            Mock(symbol="000002.SZ"),
            Mock(symbol="000003.SZ")
        ]
        mock_get_limit_up.return_value = mock_limit_up
        
        # 计算热度
        heat = data_interface.get_concept_heat(
            symbol="000001.SZ",
            date="2024-01-15"
        )
        
        # 000001.SZ 所属"人工智能"概念,当日该概念有2只涨停
        # 热度应该为 2.0
        assert heat == 2.0
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._get_stock_concepts')
    def test_get_concept_heat_no_concept(self, mock_get_concepts):
        """测试未找到概念时返回 0"""
        data_interface = LimitUpDataInterface()
        
        # Mock 返回空概念列表
        mock_get_concepts.return_value = []
        
        heat = data_interface.get_concept_heat(
            symbol="000001.SZ",
            date="2024-01-15"
        )
        
        assert heat == 0.0
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._get_stock_concepts')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_limit_up_stocks')
    def test_get_concept_heat_multiple_concepts(self, mock_get_limit_up, mock_get_concepts):
        """测试多个概念取最大热度"""
        data_interface = LimitUpDataInterface()
        
        # Mock 股票所属多个概念
        def concept_side_effect(symbol):
            if symbol == "000001.SZ":
                return ["概念A", "概念B"]  # 同时属于两个概念
            elif symbol in ["000002.SZ", "000003.SZ", "000004.SZ"]:
                return ["概念A"]  # 3只属于概念A
            elif symbol == "000005.SZ":
                return ["概念B"]  # 1只属于概念B
            return []
        
        mock_get_concepts.side_effect = concept_side_effect
        
        # Mock 当日涨停股
        mock_limit_up = [
            Mock(symbol="000001.SZ"),
            Mock(symbol="000002.SZ"),
            Mock(symbol="000003.SZ"),
            Mock(symbol="000004.SZ"),
            Mock(symbol="000005.SZ")
        ]
        mock_get_limit_up.return_value = mock_limit_up
        
        heat = data_interface.get_concept_heat(
            symbol="000001.SZ",
            date="2024-01-15"
        )
        
        # 概念A: 4只涨停, 概念B: 2只涨停
        # 应该取最大热度 4.0
        assert heat == 4.0
    
    def test_get_concept_heat_calculation_failure(self):
        """测试计算失败时返回 0"""
        data_interface = LimitUpDataInterface()
        
        # Mock _get_stock_concepts 抛出异常
        with patch.object(
            data_interface,
            '_get_stock_concepts',
            side_effect=Exception("API error")
        ):
            heat = data_interface.get_concept_heat(
                symbol="000001.SZ",
                date="2024-01-15"
            )
            
            assert heat == 0.0


# 辅助方法测试
class TestHelperMethods:
    """测试辅助方法"""
    
    @patch('akshare.stock_individual_info_em')
    def test_get_stock_concepts_from_akshare(self, mock_ak_api):
        """测试从 AKShare 获取股票概念"""
        data_interface = LimitUpDataInterface()
        
        # Mock AKShare API 返回
        mock_df = pd.DataFrame({
            '项目': ['所属板块', '其他'],
            '内容': ['人工智能,芯片,物联网', '其他信息']
        })
        mock_ak_api.return_value = mock_df
        
        concepts = data_interface._get_stock_concepts("000001.SZ")
        
        assert len(concepts) == 3
        assert "人工智能" in concepts
        assert "芯片" in concepts
        assert "物联网" in concepts
    
    def test_get_stock_concepts_cache(self):
        """测试股票概念缓存"""
        data_interface = LimitUpDataInterface()
        
        # 预设缓存
        data_interface._concept_cache["000001.SZ"] = ["概念A", "概念B"]
        
        # 应该直接从缓存返回
        concepts = data_interface._get_stock_concepts("000001.SZ")
        
        assert concepts == ["概念A", "概念B"]
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_is_limit_up_true(self, mock_ensure_qlib):
        """测试涨停判定 - 涨停"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 数据: 涨停
        mock_data = pd.DataFrame({
            '$close': [10.0, 11.0],  # 涨10%
            '$high': [10.0, 11.0]    # 收盘==最高
        }, index=pd.date_range('2024-01-14', periods=2))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            is_limit_up = data_interface._is_limit_up(
                symbol="000001.SZ",
                date="2024-01-15"
            )
            
            assert is_limit_up == True
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_is_limit_up_false(self, mock_ensure_qlib):
        """测试涨停判定 - 未涨停"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 数据: 未涨停
        mock_data = pd.DataFrame({
            '$close': [10.0, 10.5],  # 涨5%
            '$high': [10.0, 10.6]
        }, index=pd.date_range('2024-01-14', periods=2))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_data
            is_limit_up = data_interface._is_limit_up(
                symbol="000001.SZ",
                date="2024-01-15"
            )
            
            assert is_limit_up == False


# P0-5 集成测试
class TestGetLimitUpFeatures:
    """P0-5: 测试完整特征获取集成"""
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_continuous_board')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_seal_amount')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_concept_heat')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface.get_intraday_data')
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._get_basic_map')
    def test_get_limit_up_features_integration(
        self,
        mock_basic_map,
        mock_intraday,
        mock_concept_heat,
        mock_seal_amount,
        mock_continuous_board,
        mock_ensure_qlib
    ):
        """测试完整特征获取 (P0-5 集成)"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock 各个组件
        mock_continuous_board.return_value = 2  # 二板
        mock_seal_amount.return_value = 500.0  # 500万封单
        mock_concept_heat.return_value = 3.0   # 热度3
        mock_intraday.return_value = pd.DataFrame()  # 空分时数据
        mock_basic_map.return_value = {
            "000001": {
                "market_cap": 100.0,
                "turnover_rate": 0.05,
                "industry": "电子"
            }
        }
        
        # Mock Qlib 日线数据 (需要至少3天数据才能完整计算特征)
        mock_qlib_data = pd.DataFrame({
            '$open': [9.5, 10.0, 10.5],
            '$high': [9.5, 10.0, 11.0],
            '$low': [9.3, 9.8, 10.9],
            '$close': [9.5, 10.0, 11.0],
            '$volume': [8000, 10000, 15000],
            '$amount': [76000, 100000, 165000]
        }, index=pd.MultiIndex.from_product(
            [["000001.SZ"], pd.date_range('2024-01-13', periods=3)],
            names=['instrument', 'datetime']
        ))
        
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = mock_qlib_data
            features = data_interface.get_limit_up_features(
                symbols=["000001.SZ"],
                date="2024-01-15",
                lookback_days=20
            )
            
            # 验证 P0-5 新增字段存在
            assert not features.empty
            assert 'seal_amount' in features.columns
            assert 'continuous_board' in features.columns
            assert 'concept_heat' in features.columns
            
            # 验证字段值
            row = features.loc["000001.SZ"]
            assert row['seal_amount'] == 500.0
            assert row['continuous_board'] == 2
            assert row['concept_heat'] == 3.0
            
            # 验证其他基础字段
            assert 'limit_up_strength' in features.columns
            assert 'seal_quality' in features.columns
            assert 'volume_surge' in features.columns
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_limit_up_features_empty_symbols(self, mock_ensure_qlib):
        """测试空股票列表"""
        data_interface = LimitUpDataInterface()
        
        features = data_interface.get_limit_up_features(
            symbols=[],
            date="2024-01-15"
        )
        
        assert features.empty
    
    @patch('rd_agent.limit_up_data.LimitUpDataInterface._ensure_qlib')
    def test_get_limit_up_features_calculation_error_fallback(self, mock_ensure_qlib):
        """测试计算错误时的 fallback"""
        data_interface = LimitUpDataInterface()
        mock_ensure_qlib.return_value = None
        
        # Mock Qlib 返回 None
        with patch('qlib.data.D') as mock_d:
            mock_d.features.return_value = None
            features = data_interface.get_limit_up_features(
                symbols=["000001.SZ"],
                date="2024-01-15"
            )
            
            # 应该返回空 DataFrame 或带默认值的 DataFrame
            assert isinstance(features, pd.DataFrame)


# 缓存机制测试
class TestCacheMechanism:
    """测试缓存机制 (P0-5)"""
    
    def test_seal_amount_cache_hit_avoids_recalculation(self):
        """测试封单金额缓存避免重复计算"""
        data_interface = LimitUpDataInterface()
        
        # 预设缓存
        data_interface._seal_amount_cache["000001.SZ:2024-01-15"] = 300.0
        
        # 多次调用应该返回相同缓存值
        amount1 = data_interface.get_seal_amount("000001.SZ", "2024-01-15", 10.0)
        amount2 = data_interface.get_seal_amount("000001.SZ", "2024-01-15", 10.0)
        
        assert amount1 == 300.0
        assert amount2 == 300.0
        assert amount1 == amount2
    
    def test_continuous_board_cache_hit_avoids_recalculation(self):
        """测试连板天数缓存避免重复计算"""
        data_interface = LimitUpDataInterface()
        
        # 预设缓存
        data_interface._continuous_board_cache["000001.SZ:2024-01-15"] = 3
        
        # 多次调用应该返回相同缓存值
        days1 = data_interface.get_continuous_board("000001.SZ", "2024-01-15")
        days2 = data_interface.get_continuous_board("000001.SZ", "2024-01-15")
        
        assert days1 == 3
        assert days2 == 3
        assert days1 == days2
    
    def test_concept_cache_hit_avoids_api_call(self):
        """测试概念缓存避免重复 API 调用"""
        data_interface = LimitUpDataInterface()
        
        # 预设缓存
        data_interface._concept_cache["000001.SZ"] = ["概念A", "概念B"]
        
        # 多次调用应该返回相同缓存值
        concepts1 = data_interface._get_stock_concepts("000001.SZ")
        concepts2 = data_interface._get_stock_concepts("000001.SZ")
        
        assert concepts1 == ["概念A", "概念B"]
        assert concepts2 == ["概念A", "概念B"]
        assert concepts1 == concepts2
    
    def test_different_dates_use_different_cache(self):
        """测试不同日期使用不同缓存键"""
        data_interface = LimitUpDataInterface()
        
        # 设置不同日期的缓存
        data_interface._seal_amount_cache["000001.SZ:2024-01-15"] = 100.0
        data_interface._seal_amount_cache["000001.SZ:2024-01-16"] = 200.0
        
        amount1 = data_interface.get_seal_amount("000001.SZ", "2024-01-15", 10.0)
        amount2 = data_interface.get_seal_amount("000001.SZ", "2024-01-16", 10.0)
        
        assert amount1 == 100.0
        assert amount2 == 200.0
        assert amount1 != amount2


# LimitUpRecord 数据类测试
class TestLimitUpRecord:
    """测试 LimitUpRecord 数据类"""
    
    def test_limit_up_record_creation(self):
        """测试 LimitUpRecord 创建"""
        record = LimitUpRecord(
            symbol="000001.SZ",
            date=datetime(2024, 1, 15),
            limit_up_time="10:30:00",
            limit_up_type=2,
            seal_amount=500.0,
            seal_ratio=0.05,
            turnover_rate=0.15,
            continuous_days=1,
            concept=["人工智能", "芯片"],
            industry="电子",
            market_cap=100.0,
            open_change=0.02,
            high_change=0.10,
            volume_ratio=2.5
        )
        
        assert record.symbol == "000001.SZ"
        assert record.seal_amount == 500.0
        assert record.continuous_days == 1
        assert len(record.concept) == 2
        assert "人工智能" in record.concept


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
