"""
用 PyTorch 实现 `RMSNorm` 类， `SwiGLU` 模块和 `RoPE` 函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.eps)
        x_norm = x_fp32 / rms
        return x_norm.to(orig_dtype) * self.weight

class SwiGLU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        实现 SwiGLU 激活函数
        Args:
            dim_in: 输入维度
            dim_out: 输出维度
            hidden_dim: 隐藏层维度
            dtype: 参数数据类型，默认torch.bfloat16
        """
        super().__init__()
        # 都不使用 bias
        self.up_proj = nn.Linear(dim_in, hidden_dim, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(dim_in, hidden_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(hidden_dim, dim_out, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        prod_fp32 = x1.float() * F.silu(x2.float())
        return self.down_proj(prod_fp32.to(x.dtype))


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    # 生成 dim/2 个不同的频率：theta_i = theta ^ (-2i/d)
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / dim))

    # 生成位置索引m = 0,1,...,seq_len-1
    t = torch.arange(seq_len, device=device)

    # 计算m * theta_i，形状为 (seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # 计算 cos(m*theta_i) 与 sin(m*theta_i)
    freqs_cos = torch.cos(freqs.float())
    freqs_sin = torch.sin(freqs.float())

    # 返回半维度 cos/sin，形状为 (seq_len, dim/2)
    return freqs_cos.to(dtype=dtype), freqs_sin.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量，原论文使用 “偶/奇交错维度”配对旋转，
    这里使用 “前半/后半” 配对旋转。和 Hugging Face 对齐

    Args:
        x: 输入张量，形状为 (batch_size, groups/heads, seq_len, head_dim)
        freqs_cis: 预计算的频率张量，形状为 (seq_len, dim)

    Returns:
        x_rotated: 应用旋转编码后的张量，形状与输入相同
    """
    # 确保输入维度是偶数
    assert x.shape[-1] % 2 == 0, "x dimension must be even"

    orig_dtype = x.dtype
    x_fp32 = x.float()
    cos_full = torch.cat([cos_half.float(), cos_half.float()], dim=-1)
    sin_full = torch.cat([sin_half.float(), sin_half.float()], dim=-1)

    # 旋转公式：x * cos + rotate_half(x) * sin
    half = x_fp32.shape[-1] // 2
    x1 = x_fp32[..., :half]
    x2 = x_fp32[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    x_rotated = x_fp32 * cos_full + rotated * sin_full

    return x_rotated.to(orig_dtype)


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
    print(f"Freqs_cis shape: {freqs_cis[0].shape}, {freqs_cis[1].shape}")
    print(f"Freqs_cis: {freqs_cis}")

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {x.shape}")
    print(f"Input: {x}")

    # 应用旋转编码
    x_rotated = apply_rotary_emb(x, freqs_cis[0], freqs_cis[1])
    print(f"Output shape: {x_rotated.shape}")
    print(f"Output: {x_rotated}")
