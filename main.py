"""
用 PyTorch 实现 `RMSNorm` 类和 `SwiGLU` 模块
"""

from typing import Optional, Dict, List
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchview import draw_graph
from safetensors.torch import load_file
import json


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # 归一化并应用权重
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim_in, hidden_dim)
        self.w2 = nn.Linear(dim_in, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 激活函数
        x1 = self.w1(x)
        x2 = self.w2(x)
        # 使用 swiglu 函数：x1 * sigmoid(x2)
        return self.w3(x1 * F.silu(x2))


def precompute_freqs_cis(
    dim: int, seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    预计算旋转位置编码的频率张量

    Args:
        dim: 嵌入维度
        seq_len: 序列长度
        theta: 旋转位置编码的基数，默认为10000.0

    Returns:
        freqs_cis: 预计算的频率张量，形状为 (seq_len, dim)
    """
    # 计算每个维度的theta值
    # dim必须是偶数，因为旋转是2D分组的
    assert dim % 2 == 0, "dim must be even"

    # 生成dim/2个不同的theta值
    # theta_i = theta ^ (-2i/dim), i=0,1,...,dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置索引m = 0,1,...,seq_len-1
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)

    # 计算m * theta_i，形状为 (seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # 计算cos(m*theta_i)和sin(m*theta_i)，并将它们堆叠成最终的频率张量
    # 最终形状为 (seq_len, dim)
    # 对于每个位置，偶数索引是cos，奇数索引是sin
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # 将cos和sin交替堆叠，形成最终的freqs_cis
    # 例如，对于dim=4，形状变为 (seq_len, 2, 2)，然后reshape为 (seq_len, 4)
    freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=-1)
    freqs_cis = freqs_cis.reshape(seq_len, dim)

    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量

    Args:
        x: 输入张量，形状为 (batch_size, seq_len, dim)
        freqs_cis: 预计算的频率张量，形状为 (seq_len, dim)

    Returns:
        x_rotated: 应用旋转编码后的张量，形状与输入相同
    """
    # 确保输入维度是偶数
    assert x.shape[-1] % 2 == 0, "x dimension must be even"

    # 分离偶数和奇数索引的元素
    # x_even: 形状 (batch_size, seq_len, dim/2)
    # x_odd: 形状 (batch_size, seq_len, dim/2)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    # 分离freqs_cis的cos和sin部分
    # freqs_cos: 形状 (seq_len, dim/2)
    # freqs_sin: 形状 (seq_len, dim/2)
    freqs_cos = freqs_cis[..., ::2]
    freqs_sin = freqs_cis[..., 1::2]

    # 应用旋转公式：
    # x_rotated_even = x_even * cos - x_odd * sin
    # x_rotated_odd = x_even * sin + x_odd * cos
    x_rotated_even = x_even * freqs_cos - x_odd * freqs_sin
    x_rotated_odd = x_even * freqs_sin + x_odd * freqs_cos

    # 将旋转后的偶数和奇数部分重新组合
    # 使用stack和reshape来恢复原始形状
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = x_rotated.reshape(x.shape)

    return x_rotated


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, groups: Optional[int] = None, dropout: float = 0.0):
        """
        实现多头注意力机制，支持 MHA，MQA，GQA
        Args:
            hidden_size: 隐藏层维度
            heads: 头数
            groups: 分组数
            dropout: dropout概率
        """
        super().__init__()
        # 确保hidden_size是heads的倍数
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.hidden_size = hidden_size
        self.heads = heads
        if groups is None:
            groups = heads
        else:
            assert heads % groups == 0, "heads must be divisible by groups"
        self.groups = groups
        self.head_dim = hidden_size // heads
        kv_dims = self.head_dim * groups
        
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, kv_dims)
        self.w_v = nn.Linear(hidden_size, kv_dims)
        self.w_o = nn.Linear(hidden_size, hidden_size)  # 输出投影层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
        freqs_cis: 预计算的旋转位置编码，形状为 (seq_len, head_dim * 2)
        mask: 注意力掩码，形状为 (seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        """
        assert x.shape[-1] == self.hidden_size, "input hidden_size must be equal to hidden_size"
        batch_size, seq_len, hidden_size = x.shape

        q: torch.Tensor = self.w_q(x)
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        # 重塑并转置张量
        q_multi = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)
        k_multi = k.view(batch_size, seq_len, self.groups, self.head_dim).transpose(1, 2)  # (batch_size, groups, seq_len, head_dim)
        v_multi = v.view(batch_size, seq_len, self.groups, self.head_dim).transpose(1, 2)  # (batch_size, groups, seq_len, head_dim)
        
        # 应用旋转位置编码
        if freqs_cis is not None:
            q_multi = apply_rotary_emb(q_multi, freqs_cis)
            k_multi = apply_rotary_emb(k_multi, freqs_cis)
        
        # 对于GQA，将每组的k和v复制到对应的头
        k_multi = k_multi.repeat_interleave(self.heads // self.groups, dim=1)  # (batch_size, heads, seq_len, head_dim)
        v_multi = v_multi.repeat_interleave(self.heads // self.groups, dim=1)  # (batch_size, heads, seq_len, head_dim)
        
        # 计算注意力分数
        scores = q_multi @ k_multi.transpose(-2, -1) / math.sqrt(self.head_dim)  # (batch_size, heads, seq_len, seq_len)
        
        # 应用掩码
        if mask is not None:
            scores = scores + mask
        
        # 应用softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = attn_weights @ v_multi  # (batch_size, heads, seq_len, head_dim)
        
        # 合并头并应用输出投影
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
        output = self.w_o(output)  # (batch_size, seq_len, hidden_size)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        """
        实现前馈网络，包含两个线性层和 SwiGLU 激活函数
        Args:
            hidden_size: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.swiglu = SwiGLU(hidden_size * 4, hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w_1(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, groups: Optional[int] = None, dropout: float = 0.0):
        """
        实现Transformer块，包含自注意力层和前馈网络
        Args:
            hidden_size: 隐藏层维度
            heads: 头数
            groups: 分组数
            dropout: dropout概率
        """
        super().__init__()
        self.attention = Attention(hidden_size, heads, groups, dropout)
        self.ffn = FeedForward(hidden_size, dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
        freqs_cis: 预计算的旋转位置编码，形状为 (seq_len, head_dim * 2)
        mask: 注意力掩码，形状为 (seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        """
        # 自注意力层
        attn_output = self.attention(self.norm1(x), freqs_cis, mask)
        x = x + attn_output  # 残差连接
        
        # 前馈网络
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output  # 残差连接
        
        return x

class Llama(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, heads: int, groups: Optional[int] = None, dropout: float = 0.0):
        """
        实现Llama模型，包含多个Transformer块
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度
            heads: 头数
            groups: 分组数
            dropout: dropout概率
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, heads, groups, dropout)
            for _ in range(32)  # llama3-8B 有 32 个Transformer块
        ])
        self.norm = RMSNorm(hidden_size)
        self.w_o = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 输入张量，形状为 (batch_size, seq_len)
        """
        # 嵌入层
        x = self.embedding(x)
        
        # Transformer块
        for block in self.transformer_blocks:
            x = block(x)
        
        # 归一化层
        x = self.norm(x)
        
        # 输出层
        x = self.w_o(x)
        
        return x

def load_safetensors_weights(
    model: Llama,
    safetensors_path: str,
    weight_map,
    load_all: bool = True,
    layers_to_load: Optional[List[int]] = None
) -> None:
    """
    to-do fix
    加载 safetensors 权重到 Llama 模型
    
    Args:
        model: Llama 模型实例
        safetensors_path: safetensors 文件路径
        weight_map: 权重名映射字典
        load_all: 是否加载所有权重，默认为 True
        layers_to_load: 如果 load_all 为 False，指定要加载的层索引列表
    """
    # 读取 safetensors 文件
    print(f"Loading weights from {safetensors_path}...")
    safetensors_weights = load_file(safetensors_path)
    
    # 准备要加载的权重
    state_dict = {}
    for safetensor_key, weight in safetensors_weights.items():
        if safetensor_key not in weight_map:
            print(f"Warning: {safetensor_key} not in weight map, skipping...")
            continue
        
        # 检查是否需要加载该层
        should_load = load_all
        if not load_all:
            # 检查是否是层相关的权重
            if ".layers." in safetensor_key:
                # 提取层索引
                layer_idx = int(safetensor_key.split(".layers.")[1].split(".")[0])
                if layer_idx not in layers_to_load:
                    continue
            should_load = True
        
        if should_load:
            # 获取自定义模型的权重名
            custom_key = weight_map[safetensor_key]
            state_dict[custom_key] = weight
    
    # 加载权重到模型
    print(f"Loading {len(state_dict)} weights into model...")
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded successfully!")


def test_llama():
    with torch.device("meta"):
        # 创建模型
        model = Llama(vocab_size=128000, hidden_size=4096, heads=32, groups=8)
    meta_input_ids = torch.randint(0, 10000, (1, 128), device="meta")
    model_graph = draw_graph(
        model, 
        input_data=meta_input_ids,
        expand_nested=True,
        depth=1,  # 2/3
        device="meta" 
    )
    filename = "llama3_structure_meta"
    model_graph.visual_graph.render(filename, format="png")
    model_path = '/Volumes/My Passport/model/Llama-3-8B'
    index_path = model_path + '/model.safetensors.index.json'
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
    load_safetensors_weights(model, model_path, weight_map, False, [0])

def main():
    print("Hello from nano-llama-3!")

    # 测试 RMSNorm
    print("\nTesting RMSNorm:")
    rms_norm = RMSNorm(dim=10)
    x = torch.randn(3, 10)
    print(f"Input shape: {x.shape}")
    output = rms_norm(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # 测试 SwiGLU
    print("\nTesting SwiGLU:")
    swiglu = SwiGLU(dim_in=10, dim_out=5, hidden_dim=20)
    x = torch.randn(3, 10)
    print(f"Input shape: {x.shape}")
    output = swiglu(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # 测试旋转位置编码
    print("\nTesting Rotary Position Embedding:")
    batch_size = 2
    seq_len = 5
    dim = 4

    # 预计算频率
    freqs_cis = precompute_freqs_cis(dim=dim, seq_len=seq_len)
    print(f"Freqs_cis shape: {freqs_cis.shape}")
    print(f"Freqs_cis: {freqs_cis}")

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {x.shape}")
    print(f"Input: {x}")

    # 应用旋转编码
    x_rotated = apply_rotary_emb(x, freqs_cis)
    print(f"Output shape: {x_rotated.shape}")
    print(f"Output: {x_rotated}")

    # 测试注意力模块
    print("\nTesting Attention:")
    attention = Attention(hidden_size=12, heads=4, groups=2)
    x = torch.randn(3, 5, 12)
    print(f"Input shape: {x.shape}")
    output = attention(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    test_llama()





def main():
    print("Hello from nano-llama-3!")

    # 测试 RMSNorm
    print("\nTesting RMSNorm:")
    rms_norm = RMSNorm(dim=10)
    x = torch.randn(3, 10)
    print(f"Input shape: {x.shape}")
    output = rms_norm(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # 测试 SwiGLU
    print("\nTesting SwiGLU:")
    swiglu = SwiGLU(dim_in=10, dim_out=5, hidden_dim=20)
    x = torch.randn(3, 10)
    print(f"Input shape: {x.shape}")
    output = swiglu(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # 测试旋转位置编码
    print("\nTesting Rotary Position Embedding:")
    batch_size = 2
    seq_len = 5
    dim = 4

    # 预计算频率
    freqs_cis = precompute_freqs_cis(dim=dim, seq_len=seq_len)
    print(f"Freqs_cis shape: {freqs_cis.shape}")
    print(f"Freqs_cis: {freqs_cis}")

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {x.shape}")
    print(f"Input: {x}")

    # 应用旋转编码
    x_rotated = apply_rotary_emb(x, freqs_cis)
    print(f"Output shape: {x_rotated.shape}")
    print(f"Output: {x_rotated}")

    # 测试注意力模块
    print("\nTesting Attention:")
    attention = Attention(hidden_size=12, heads=4, groups=2)
    x = torch.randn(3, 5, 12)
    print(f"Input shape: {x.shape}")
    output = attention(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    test_llama()


if __name__ == "__main__":
    main()
