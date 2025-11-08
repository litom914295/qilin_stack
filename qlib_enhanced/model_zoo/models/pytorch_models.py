"""
PyTorch神经网络模型实现 - 用于Qlib Model Zoo
包含: MLP, LSTM, GRU, ALSTM, Transformer, TCN等
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch未安装，请运行: pip install torch")


class BaseNNModel:
    """神经网络模型基类"""
    
    def __init__(self, 
                 lr: float = 0.001,
                 n_epochs: int = 100,
                 batch_size: int = 512,
                 early_stop: int = 20,
                 device: str = None,
                 **kwargs):
        """
        初始化神经网络模型
        
        Args:
            lr: 学习率
            n_epochs: 训练轮数
            batch_size: 批次大小
            early_stop: 早停轮数
            device: 设备 ('cpu', 'cuda' 或 None自动选择)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch未安装")
        
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        
        # 选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, 
            train_data: Union[pd.DataFrame, tuple],
            valid_data: Optional[Union[pd.DataFrame, tuple]] = None,
            **kwargs):
        """训练模型"""
        # 解析数据
        X_train, y_train = self._parse_data(train_data)
        
        # 特征维度
        input_dim = X_train.shape[1]
        self.feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        
        # 创建模型
        if self.model is None:
            self.model = self._create_network(input_dim)
            self.model.to(self.device)
        
        # 创建数据加载器
        X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
        y_train_tensor = torch.FloatTensor(y_train.values if isinstance(y_train, pd.Series) else y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 验证集
        if valid_data is not None:
            X_valid, y_valid = self._parse_data(valid_data)
            X_valid_tensor = torch.FloatTensor(X_valid.values if isinstance(X_valid, pd.DataFrame) else X_valid)
            y_valid_tensor = torch.FloatTensor(y_valid.values if isinstance(y_valid, pd.Series) else y_valid)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"开始训练{self.__class__.__name__}，设备: {self.device}")
        
        for epoch in range(self.n_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            if valid_data is not None:
                self.model.eval()
                with torch.no_grad():
                    X_valid_device = X_valid_tensor.to(self.device)
                    y_valid_device = y_valid_tensor.to(self.device)
                    
                    valid_outputs = self.model(X_valid_device)
                    valid_loss = criterion(valid_outputs.squeeze(), y_valid_device).item()
                
                # 早停检查
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stop:
                    self.logger.info(f"早停触发，epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss:.4f}")
        
        self.logger.info(f"{self.__class__.__name__}训练完成")
        return self
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        
        # 解析数据
        if isinstance(data, pd.DataFrame):
            if 'LABEL0' in data.columns:
                X = data.drop(columns=['LABEL0'])
            else:
                X = data
            index = data.index
            X_array = X.values
        else:
            X_array = data
            index = None
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()
        
        # 返回Series格式
        if index is not None:
            return pd.Series(predictions, index=index)
        else:
            return pd.Series(predictions)
    
    def _parse_data(self, data: Union[pd.DataFrame, tuple]):
        """解析数据格式"""
        if isinstance(data, tuple):
            X, y = data
            return X, y
        elif isinstance(data, pd.DataFrame):
            if 'LABEL0' in data.columns:
                y = data['LABEL0']
                X = data.drop(columns=['LABEL0'])
            else:
                y = data.iloc[:, -1]
                X = data.iloc[:, :-1]
            return X, y
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
    def _create_network(self, input_dim: int):
        """创建网络结构 - 子类需要实现"""
        raise NotImplementedError("子类需要实现_create_network方法")


class MLPModel(BaseNNModel):
    """多层感知机 (MLP)"""
    
    def __init__(self, 
                 hidden_size: int = 128,
                 dropout: float = 0.3,
                 num_layers: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
    
    def _create_network(self, input_dim: int):
        """创建MLP网络"""
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # 隐藏层
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        # 输出层
        layers.append(nn.Linear(self.hidden_size, 1))
        
        return nn.Sequential(*layers)


class LSTMModel(BaseNNModel):
    """LSTM模型"""
    
    def __init__(self, 
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
    
    def _create_network(self, input_dim: int):
        """创建LSTM网络"""
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, 
                    hidden_size, 
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                # x shape: (batch, features)
                # 添加时间维度: (batch, 1, features)
                x = x.unsqueeze(1)
                
                # LSTM forward
                lstm_out, _ = self.lstm(x)
                
                # 取最后一个时间步
                last_out = lstm_out[:, -1, :]
                
                # Dropout and FC
                out = self.dropout(last_out)
                out = self.fc(out)
                
                return out
        
        return LSTMNet(input_dim, self.hidden_size, self.num_layers, self.dropout_rate)


class GRUModel(BaseNNModel):
    """GRU模型"""
    
    def __init__(self, 
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
    
    def _create_network(self, input_dim: int):
        """创建GRU网络"""
        class GRUNet(nn.Module):
            def __init__(self, input_dim, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    input_dim, 
                    hidden_size, 
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                x = x.unsqueeze(1)
                gru_out, _ = self.gru(x)
                last_out = gru_out[:, -1, :]
                out = self.dropout(last_out)
                out = self.fc(out)
                return out
        
        return GRUNet(input_dim, self.hidden_size, self.num_layers, self.dropout_rate)


class TransformerModel(BaseNNModel):
    """Transformer模型"""
    
    def __init__(self, 
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers_count = num_layers
        self.dropout_rate = dropout
    
    def _create_network(self, input_dim: int):
        """创建Transformer网络"""
        class TransformerNet(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
                super().__init__()
                self.embedding = nn.Linear(input_dim, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model*4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(d_model, 1)
            
            def forward(self, x):
                # Embedding
                x = x.unsqueeze(1)  # (batch, 1, features)
                x = self.embedding(x)  # (batch, 1, d_model)
                
                # Transformer
                x = self.transformer(x)
                
                # 全局平均池化
                x = x.mean(dim=1)
                
                # FC
                x = self.dropout(x)
                out = self.fc(x)
                return out
        
        return TransformerNet(input_dim, self.d_model, self.nhead, self.num_layers_count, self.dropout_rate)


class TCNModel(BaseNNModel):
    """时间卷积网络 (TCN)"""
    
    def __init__(self, 
                 num_channels: List[int] = None,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels or [32, 32, 32]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
    
    def _create_network(self, input_dim: int):
        """创建TCN网络"""
        class TCNNet(nn.Module):
            def __init__(self, input_dim, num_channels, kernel_size, dropout):
                super().__init__()
                
                layers = []
                num_levels = len(num_channels)
                
                for i in range(num_levels):
                    in_channels = input_dim if i == 0 else num_channels[i-1]
                    out_channels = num_channels[i]
                    
                    # 因果卷积
                    dilation = 2 ** i
                    padding = (kernel_size - 1) * dilation
                    
                    layers.append(nn.Conv1d(
                        in_channels, 
                        out_channels, 
                        kernel_size,
                        padding=padding,
                        dilation=dilation
                    ))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                
                self.network = nn.Sequential(*layers)
                self.fc = nn.Linear(num_channels[-1], 1)
            
            def forward(self, x):
                # x: (batch, features)
                # TCN expects (batch, channels, seq_len)
                x = x.unsqueeze(2)  # (batch, features, 1)
                
                # TCN forward
                x = self.network(x)
                
                # 全局平均池化
                x = x.mean(dim=2)  # (batch, channels)
                
                # FC
                out = self.fc(x)
                return out
        
        return TCNNet(input_dim, self.num_channels, self.kernel_size, self.dropout_rate)


# ALSTM (Attention LSTM) 简化版本
class ALSTMModel(LSTMModel):
    """注意力LSTM - 继承LSTM并添加注意力机制"""
    
    def _create_network(self, input_dim: int):
        """创建ALSTM网络"""
        class ALSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, 
                    hidden_size, 
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                
                # 注意力层
                self.attention = nn.Linear(hidden_size, 1)
                
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                x = x.unsqueeze(1)
                
                # LSTM
                lstm_out, _ = self.lstm(x)
                
                # 注意力权重
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                
                # 加权和
                context = torch.sum(attn_weights * lstm_out, dim=1)
                
                # FC
                out = self.dropout(context)
                out = self.fc(out)
                
                return out
        
        return ALSTMNet(input_dim, self.hidden_size, self.num_layers, self.dropout_rate)
