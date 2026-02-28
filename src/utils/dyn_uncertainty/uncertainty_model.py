import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.time_utils import get_embedder
import numpy as np
class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, output_dim: int = 1, 
                 net_depth: int = 2, net_activation=F.relu, weight_init: str = 'he_uniform', time_dim: int = 0):
        super(MLPNetwork, self).__init__()
        
        self.output_layer_input_dim = hidden_dim

        self.embed_time_fn = None
        if time_dim > 0:
            self.embed_time_fn, self.time_input_ch = get_embedder(10, 1)
            input_dim += self.time_input_ch

        # Initialize MLP layers
        self.layers = nn.ModuleList()
        for i in range(net_depth):
            dense_layer = nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            
            # Apply weight initialization
            if weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(dense_layer.weight, nonlinearity='relu')
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(dense_layer.weight)
            else:
                raise NotImplementedError(f"Unknown Weight initialization method {weight_init}")

            self.layers.append(dense_layer)
        
        # Initialize output layer
        self.output_layer = nn.Linear(self.output_layer_input_dim, output_dim)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        
        # Set activation function
        self.net_activation = net_activation
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        # Get input dimensions
        H, W, C = x.shape[-3:]
        input_with_batch_dim = True
        
        # Add batch dimension if not present
        if len(x.shape) == 3:
            input_with_batch_dim = False
            x = x.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # Flatten input for MLP
        x = x.view(-1, x.size()[-1])
        
        if isinstance(t, (int, float)):
            t = torch.tensor([t])  # 标量转为一维张量
        elif isinstance(t, (list, tuple, range)):
            t = torch.tensor(list(t))  # 转换为张量（先转 list 再转 tensor）
        elif isinstance(t, np.ndarray):
            t = torch.from_numpy(t)  # NumPy 数组转张量

        if self.embed_time_fn is not None and t is not None:  # 检查是否存在时间编码
            # 确保t的形状正确
            if isinstance(t, (int, float)) or (isinstance(t, torch.Tensor) and t.numel() == 1):
                t = torch.full((x.shape[0], 1), float(t), device=x.device)
            elif len(t.shape) == 1:
                t = t.unsqueeze(-1)

                hw_size = H * W
        
                # 扩展t以匹配x的形状
                # 首先扩展t为 [batch, 1, 1]
                t = t.unsqueeze(-1)
                # 然后扩展为 [batch, H*W, 1]
                t = t.expand(-1, hw_size, -1)
                # 最后展平为 [batch*H*W, 1]
                t = t.reshape(-1, 1)

            if isinstance(t, torch.Tensor) and t.device != x.device:
                t = t.to(x.device)

            # 对时间进行编码
            t_emb = self.embed_time_fn(t)
            # 如果需要，扩展时间编码以匹配x的批次大小
            if t_emb.shape[0] != x.shape[0]:
                t_emb = t_emb.expand(x.shape[0], -1)
            # 将时间编码与特征拼接
            x = torch.cat([x, t_emb], dim=-1)

        # Pass through MLP layers
        for layer in self.layers:
            x = layer(x)
            x = self.net_activation(x)
            x = F.dropout(x, p=0.2)

        # Pass through output layer and apply softplus activation
        x = self.output_layer(x)
        x = self.softplus(x)

        # Reshape output to original dimensions
        if input_with_batch_dim:
            x = x.view(batch_size, H, W)
        else:
            x = x.view(H, W)

        return x

def generate_uncertainty_mlp(n_features: int, time_dim: int = 0) -> MLPNetwork:
    # Create and return an MLP network with the specified input dimensions
    network = MLPNetwork(input_dim=n_features, time_dim=time_dim).cuda()
    return network