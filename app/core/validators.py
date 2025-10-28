"""
输入验证模块
提供统一的数据验证和清洗功能
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from enum import Enum

class ValidationError(Exception):
    """验证错误异常"""
    pass

class MarketType(Enum):
    """市场类型枚举"""
    CN_STOCK = "cn_stock"  # A股
    US_STOCK = "us_stock"  # 美股
    HK_STOCK = "hk_stock"  # 港股
    CRYPTO = "crypto"      # 加密货币

class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "BUY"
    SELL = "SELL"

class Validator:
    """通用验证器"""
    
    # 股票代码正则表达式
    SYMBOL_PATTERNS = {
        MarketType.CN_STOCK: r'^(SH|SZ|BJ)\d{6}$',  # 沪深京
        MarketType.US_STOCK: r'^[A-Z]{1,5}$',        # 美股
        MarketType.HK_STOCK: r'^\d{5}$',             # 港股
        MarketType.CRYPTO: r'^[A-Z]{2,10}$'          # 加密货币
    }
    
    # 交易限制
    MAX_ORDER_QTY = 1000000  # 最大订单数量
    MIN_ORDER_QTY = 100      # 最小订单数量（A股1手）
    MAX_PRICE = 10000        # 最大价格
    MIN_PRICE = 0.01         # 最小价格
    MAX_POSITION_RATIO = 0.3 # 最大仓位比例
    
    @classmethod
    def normalize_symbol(cls, symbol: str, output_format: str = "qlib") -> str:
        """
        标准化股票代码格式
        支持多种格式的转换: 600000.SH <-> SH600000
        
        Args:
            symbol: 股票代码 (支持 600000.SH 或 SH600000 格式)
            output_format: 输出格式 ("qlib" = SH600000, "standard" = 600000.SH)
            
        Returns:
            标准化的股票代码
            
        Examples:
            >>> normalize_symbol("600000.SH", "qlib")
            "SH600000"
            >>> normalize_symbol("SH600000", "standard")
            "600000.SH"
        """
        if not symbol:
            raise ValidationError("股票代码不能为空")
        
        symbol = str(symbol).upper().strip()
        
        # 验证基本格式 - 必须包含数字
        if not re.search(r'\d', symbol):
            raise ValidationError(f"无效的股票代码: {symbol} (必须包含数字)")
        
        # 检测当前格式
        if '.' in symbol:  # 格式: 600000.SH
            parts = symbol.split('.')
            if len(parts) != 2:
                raise ValidationError(f"无效的股票代码格式: {symbol}")
            code, exchange = parts
            
            # 验证代码是6位数字
            if not (code.isdigit() and len(code) == 6):
                raise ValidationError(f"无效的股票代码: {symbol} (代码部分应为6位数字)")
            
            # 标准化交易所代码
            exchange_map = {'SH': 'SH', 'SZ': 'SZ', 'BJ': 'BJ', 
                          'SS': 'SH', 'XSHG': 'SH', 'XSHE': 'SZ'}
            exchange = exchange_map.get(exchange, exchange)
            
            if output_format == "qlib":
                return f"{exchange}{code}"
            else:
                return f"{code}.{exchange}"
        
        elif len(symbol) >= 2 and symbol[:2] in ['SH', 'SZ', 'BJ']:  # 格式: SH600000
            exchange = symbol[:2]
            code = symbol[2:]
            
            # 验证代码部分是6位数字
            if not (code.isdigit() and len(code) == 6):
                raise ValidationError(f"无效的股票代码: {symbol} (代码部分应为6位数字)")
            
            if output_format == "qlib":
                return f"{exchange}{code}"
            else:
                return f"{code}.{exchange}"
        
        else:
            # 无交易所前缀,验证并自动识别
            # 必须是6位数字
            if not (symbol.isdigit() and len(symbol) == 6):
                raise ValidationError(f"无效的股票代码: {symbol} (应为6位数字或包含交易所前缀)")
            
            if symbol.startswith('6'):
                exchange = 'SH'
            elif symbol.startswith(('0', '3')):
                exchange = 'SZ'
            elif symbol.startswith(('4', '8')):
                exchange = 'BJ'
            else:
                raise ValidationError(f"无法识别的股票代码格式: {symbol}")
            
            if output_format == "qlib":
                return f"{exchange}{symbol}"
            else:
                return f"{symbol}.{exchange}"
    
    @classmethod
    def validate_symbol(cls, symbol: str, market: MarketType = MarketType.CN_STOCK) -> str:
        """
        验证股票代码
        
        Args:
            symbol: 股票代码
            market: 市场类型
            
        Returns:
            标准化的股票代码
            
        Raises:
            ValidationError: 验证失败
        """
        if not symbol:
            raise ValidationError("股票代码不能为空")
        
        if not isinstance(symbol, str):
            raise ValidationError(f"股票代码必须是字符串类型，当前类型: {type(symbol)}")
        
        # 先标准化为qlib格式
        try:
            symbol = cls.normalize_symbol(symbol, "qlib")
        except ValidationError:
            # 如果标准化失败,使用原始格式
            symbol = symbol.upper().strip()
        
        # 验证格式
        pattern = cls.SYMBOL_PATTERNS.get(market)
        if pattern and not re.match(pattern, symbol):
            raise ValidationError(
                f"无效的股票代码格式: {symbol}，"
                f"预期格式: {pattern}"
            )
        
        return symbol
    @classmethod
    def validate_quantity(cls, qty: Union[int, float]) -> int:
        """
        验证交易数量
        
        Args:
            qty: 交易数量
            
        Returns:
            验证后的数量
            
        Raises:
            ValidationError: 验证失败
        """
        try:
            qty = int(qty)
        except (ValueError, TypeError):
            raise ValidationError(f"无效的数量: {qty}")
        
        if qty <= 0:
            raise ValidationError(f"数量必须大于0，当前值: {qty}")
        
        if qty > cls.MAX_ORDER_QTY:
            raise ValidationError(f"数量超过最大限制 {cls.MAX_ORDER_QTY}，当前值: {qty}")
        
        if qty < cls.MIN_ORDER_QTY:
            raise ValidationError(f"数量低于最小限制 {cls.MIN_ORDER_QTY}，当前值: {qty}")
        
        # A股必须是100的整数倍
        if qty % 100 != 0:
            raise ValidationError(f"A股交易数量必须是100的整数倍，当前值: {qty}")
        
        return qty
    
    @classmethod
    def validate_price(cls, price: Union[float, int], symbol: str = None) -> float:
        """
        验证价格
        
        Args:
            price: 价格
            symbol: 股票代码（用于特定规则）
            
        Returns:
            验证后的价格
            
        Raises:
            ValidationError: 验证失败
        """
        try:
            price = float(price)
        except (ValueError, TypeError):
            raise ValidationError(f"无效的价格: {price}")
        
        if price <= 0:
            raise ValidationError(f"价格必须大于0，当前值: {price}")
        
        if price > cls.MAX_PRICE:
            raise ValidationError(f"价格超过最大限制 {cls.MAX_PRICE}，当前值: {price}")
        
        if price < cls.MIN_PRICE:
            raise ValidationError(f"价格低于最小限制 {cls.MIN_PRICE}，当前值: {price}")
        
        # A股价格精度（保留2位小数）
        price = round(price, 2)
        
        return price
    
    @classmethod
    def validate_order(cls, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证订单数据
        
        Args:
            order: 订单字典
            
        Returns:
            验证后的订单
            
        Raises:
            ValidationError: 验证失败
        """
        validated_order = {}
        
        # 验证必填字段
        required_fields = ['symbol', 'side', 'quantity', 'price']
        for field in required_fields:
            if field not in order:
                raise ValidationError(f"缺少必填字段: {field}")
        
        # 验证股票代码
        validated_order['symbol'] = cls.validate_symbol(order['symbol'])
        
        # 验证方向
        side = order['side'].upper()
        if side not in ['BUY', 'SELL']:
            raise ValidationError(f"无效的订单方向: {side}")
        validated_order['side'] = side
        
        # 验证数量
        validated_order['quantity'] = cls.validate_quantity(order['quantity'])
        
        # 验证价格
        validated_order['price'] = cls.validate_price(
            order['price'], 
            validated_order['symbol']
        )
        
        # 验证订单类型（如果提供）
        if 'order_type' in order:
            order_type = order['order_type'].upper()
            if order_type not in ['MARKET', 'LIMIT', 'STOP']:
                raise ValidationError(f"无效的订单类型: {order_type}")
            validated_order['order_type'] = order_type
        else:
            validated_order['order_type'] = 'LIMIT'  # 默认限价单
        
        # 验证时间戳
        if 'timestamp' in order:
            try:
                if isinstance(order['timestamp'], str):
                    validated_order['timestamp'] = datetime.fromisoformat(order['timestamp'])
                elif isinstance(order['timestamp'], datetime):
                    validated_order['timestamp'] = order['timestamp']
                else:
                    validated_order['timestamp'] = datetime.now()
            except Exception:
                validated_order['timestamp'] = datetime.now()
        else:
            validated_order['timestamp'] = datetime.now()
        
        return validated_order
    
    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """
        验证DataFrame数据
        
        Args:
            df: 数据框
            required_columns: 必需的列
            
        Returns:
            验证后的数据框
            
        Raises:
            ValidationError: 验证失败
        """
        if df is None or df.empty:
            raise ValidationError("数据框为空")
        
        # 检查必需的列
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValidationError(f"缺少必需的列: {missing_cols}")
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValidationError(f"列 {col} 必须是数值类型")
        
        # 检查数据有效性
        if 'high' in df.columns and 'low' in df.columns:
            invalid_rows = df[df['high'] < df['low']]
            if not invalid_rows.empty:
                raise ValidationError(f"发现无效数据：最高价低于最低价，行数: {len(invalid_rows)}")
        
        # 检查空值
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            raise ValidationError(f"发现空值: {null_counts[null_counts > 0].to_dict()}")
        
        # 检查极端值
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                raise ValidationError("发现无效的收盘价（<=0）")
            
            # 检查异常涨跌幅（超过20%为涨跌停）
            if len(df) > 1:
                returns = df['close'].pct_change()
                extreme_returns = returns[abs(returns) > 0.2]
                if len(extreme_returns) > len(df) * 0.1:  # 超过10%的数据异常
                    raise ValidationError(f"发现过多的极端涨跌幅数据: {len(extreme_returns)}条")
        
        return df
    
    @classmethod
    def validate_parameter(cls, param_name: str, value: Any, 
                         min_val: Optional[Union[int, float]] = None,
                         max_val: Optional[Union[int, float]] = None,
                         allowed_values: Optional[List[Any]] = None) -> Any:
        """
        配置驱动的参数验证
        
        Args:
            param_name: 参数名称
            value: 参数值
            min_val: 最小值
            max_val: 最大值
            allowed_values: 允许的值列表
            
        Returns:
            验证后的值
            
        Raises:
            ValidationError: 验证失败
        """
        if value is None:
            raise ValidationError(f"参数 {param_name} 不能为空")
        
        # 检查允许值列表
        if allowed_values is not None:
            if value not in allowed_values:
                raise ValidationError(
                    f"参数 {param_name} 的值 {value} 不在允许的值列表中: {allowed_values}"
                )
            return value
        
        # 数值范围验证
        if isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                raise ValidationError(
                    f"参数 {param_name} 的值 {value} 小于最小值 {min_val}"
                )
            if max_val is not None and value > max_val:
                raise ValidationError(
                    f"参数 {param_name} 的值 {value} 大于最大值 {max_val}"
                )
        
        return value
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], 
                       config_schema: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        验证配置文件 (支持配置驱动的验证)
        
        Args:
            config: 配置字典
            config_schema: 配置模式定义
                例如: {
                    'topk': {'min': 1, 'max': 10, 'type': int},
                    'max_runtime_sec': {'min': 10, 'max': 300, 'type': int}
                }
            
        Returns:
            验证后的配置
            
        Raises:
            ValidationError: 验证失败
        """
        # 如果提供了配置模式,按模式验证
        if config_schema:
            # 先检查必填字段
            if not config:
                required_keys = [k for k, v in config_schema.items() if v.get('required', False)]
                if required_keys:
                    raise ValidationError(f"缺少必需的配置项: {', '.join(required_keys)}")
                else:
                    raise ValidationError("配置不能为空")
            
            for key, schema in config_schema.items():
                if key in config:
                    value = config[key]
                    
                    # 类型检查
                    expected_type = schema.get('type')
                    if expected_type and not isinstance(value, expected_type):
                        raise ValidationError(
                            f"参数 {key} 的类型错误: 期望 {expected_type}, 实际 {type(value)}"
                        )
                    
                    # 范围检查
                    min_val = schema.get('min')
                    max_val = schema.get('max')
                    allowed = schema.get('allowed_values')
                    
                    config[key] = cls.validate_parameter(
                        key, value, min_val, max_val, allowed
                    )
                elif schema.get('required', False):
                    raise ValidationError(f"缺少必需的配置项: {key}")
                elif 'default' in schema:
                    config[key] = schema['default']
            
            return config
        
        # 默认验证逻辑
        # 验证必需的配置项
        required_keys = ['market', 'capital']
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"缺少必需的配置项: {key}")
        
        # 验证资金
        capital = config.get('capital', 0)
        if not isinstance(capital, (int, float)) or capital <= 0:
            raise ValidationError(f"无效的资金配置: {capital}")
        
        # 验证市场类型
        market = config.get('market', '').lower()
        valid_markets = ['cn', 'us', 'hk']
        if market not in valid_markets:
            raise ValidationError(f"无效的市场类型: {market}，有效值: {valid_markets}")
        
        # 验证风控参数
        if 'risk' in config:
            risk_config = config['risk']
            
            # 最大仓位
            max_position = risk_config.get('max_position_ratio', 1.0)
            if not 0 < max_position <= 1:
                raise ValidationError(f"无效的最大仓位比例: {max_position}")
            
            # 止损比例
            stop_loss = risk_config.get('stop_loss_ratio', 0)
            if stop_loss < 0 or stop_loss > 0.5:
                raise ValidationError(f"无效的止损比例: {stop_loss}")
        
        return config
    
    @classmethod
    def sanitize_input(cls, input_str: str, max_length: int = 100) -> str:
        """
        清理和净化输入字符串
        
        Args:
            input_str: 输入字符串
            max_length: 最大长度
            
        Returns:
            清理后的字符串
        """
        if not input_str:
            return ""
        
        # 转换为字符串
        input_str = str(input_str)
        
        # 移除危险字符
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '\\', '`', '$', '|']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')
        
        # 移除控制字符
        input_str = ''.join(char for char in input_str if ord(char) >= 32)
        
        # 限制长度
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # 去除首尾空白
        input_str = input_str.strip()
        
        return input_str


class RiskValidator:
    """风险验证器"""
    
    @classmethod
    def validate_position_size(cls, position: float, capital: float, max_ratio: float = 0.3) -> bool:
        """
        验证仓位大小
        
        Args:
            position: 仓位金额
            capital: 总资金
            max_ratio: 最大仓位比例
            
        Returns:
            是否通过验证
        """
        if capital <= 0:
            raise ValidationError("资金必须大于0")
        
        position_ratio = position / capital
        
        if position_ratio > max_ratio:
            raise ValidationError(
                f"仓位比例 {position_ratio:.2%} 超过最大限制 {max_ratio:.2%}"
            )
        
        return True
    
    @classmethod
    def validate_stop_loss(cls, entry_price: float, stop_price: float, max_loss: float = 0.05) -> bool:
        """
        验证止损设置
        
        Args:
            entry_price: 入场价格
            stop_price: 止损价格
            max_loss: 最大亏损比例
            
        Returns:
            是否通过验证
        """
        if entry_price <= 0 or stop_price <= 0:
            raise ValidationError("价格必须大于0")
        
        loss_ratio = abs(stop_price - entry_price) / entry_price
        
        if loss_ratio > max_loss:
            raise ValidationError(
                f"止损比例 {loss_ratio:.2%} 超过最大限制 {max_loss:.2%}"
            )
        
        return True
    
    @classmethod
    def validate_leverage(cls, leverage: float, max_leverage: float = 2.0) -> bool:
        """
        验证杠杆率
        
        Args:
            leverage: 杠杆倍数
            max_leverage: 最大杠杆
            
        Returns:
            是否通过验证
        """
        if leverage < 1:
            raise ValidationError("杠杆率必须大于等于1")
        
        if leverage > max_leverage:
            raise ValidationError(
                f"杠杆率 {leverage} 超过最大限制 {max_leverage}"
            )
        
        return True
