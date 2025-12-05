"""
用 PyTorch 实现 `RMSNorm` 类， `SwiGLU` 模块和 `RoPE` 函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """
        实现 SwiGLU 激活函数
        Args:
            dim_in: 输入维度
            dim_out: 输出维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        # 都不使用 bias
        self.up_proj = nn.Linear(dim_in, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(dim_in, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 激活函数
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        # 使用 swiglu 函数：x1 * sigmoid(x2)
        return self.down_proj(x1 * F.silu(x2))


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