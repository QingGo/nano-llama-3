"""
用 PyTorch 实现 `Attention`，`TransformerBlock` 类
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import RMSNorm, SwiGLU, apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        groups: Optional[int] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        实现多头注意力机制，支持 MHA，MQA，GQA
        Args:
            hidden_size: 隐藏层维度
            heads: 头数
            groups: 分组数
            dropout: dropout概率
            dtype: 参数数据类型，默认torch.bfloat16
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

        # 注意，这里投影都不需要 bias，避免对 RoPE 进行干扰，且 RMSNorm 也证明 bias 不重要
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.w_k = nn.Linear(hidden_size, kv_dims, bias=False, dtype=dtype)
        self.w_v = nn.Linear(hidden_size, kv_dims, bias=False, dtype=dtype)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)  # 输出投影层
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
        freqs_cis: 预计算的旋转位置编码，形状为 (seq_len, head_dim * 2)
        mask: 注意力掩码，形状为 (seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        """
        assert x.shape[-1] == self.hidden_size, (
            "input hidden_size must be equal to hidden_size"
        )
        batch_size, seq_len, hidden_size = x.shape

        q: torch.Tensor = self.w_q(x)
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        # 重塑并转置张量
        q_multi = q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, heads, seq_len, head_dim)
        k_multi = k.view(batch_size, seq_len, self.groups, self.head_dim).transpose(
            1, 2
        )  # (batch_size, groups, seq_len, head_dim)
        v_multi = v.view(batch_size, seq_len, self.groups, self.head_dim).transpose(
            1, 2
        )  # (batch_size, groups, seq_len, head_dim)

        # 应用旋转位置编码
        if freqs_cis is not None:
            cos_half, sin_half = freqs_cis
            q_multi = apply_rotary_emb(q_multi, cos_half, sin_half)
            k_multi = apply_rotary_emb(k_multi, cos_half, sin_half)

        # 对于GQA，将每组的k和v复制到对应的头
        k_multi = k_multi.repeat_interleave(
            self.heads // self.groups, dim=1
        )  # (batch_size, heads, seq_len, head_dim)
        v_multi = v_multi.repeat_interleave(
            self.heads // self.groups, dim=1
        )  # (batch_size, heads, seq_len, head_dim)

        q_multi = q_multi.float()
        k_multi = k_multi.float()
        v_multi = v_multi.float()
        spda_dropout = self.dropout.p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q_multi,
            k_multi,
            v_multi,
            attn_mask=None,
            dropout_p=spda_dropout,
            is_causal=True,
        )

        # 合并头并应用输出投影
        output = output.transpose(1, 2).reshape(
            batch_size, seq_len, hidden_size
        )  # (batch_size, seq_len, hidden_size)
        output = self.w_o(output.to(x.dtype))  # (batch_size, seq_len, hidden_size)

        return output


class FeedForward(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16):
        """
        实现前馈网络，只包含 SwiGLU 激活函数
        Args:
            hidden_size: 隐藏层维度
            dropout: dropout概率
            dtype: 参数数据类型，默认torch.bfloat16
        """
        super().__init__()
        # 注意这里 SwiGLU 输入输出维度都是 input_size，中间升维成 hidden_size 和门控逐元素相乘
        self.swiglu = SwiGLU(input_size, input_size, hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swiglu(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        heads: int,
        groups: Optional[int] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        实现 Transformer 块
        Args:
            hidden_size: 隐藏层维度
            ffn_hidden_size: 前馈网络隐藏层维度
            heads: 头数
            groups: 分组数
            dropout: dropout概率
            dtype: 参数数据类型，默认torch.bfloat16
        """
        super().__init__()
        self.attention = Attention(hidden_size, heads, groups, dropout, dtype=dtype)
        self.ffn = FeedForward(hidden_size, ffn_hidden_size, dtype=dtype)
        self.norm1 = RMSNorm(hidden_size, eps=1e-5, dtype=dtype)
        self.norm2 = RMSNorm(hidden_size, eps=1e-5, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
        freqs_cis: 预计算的旋转位置编码，形状为 (seq_len, head_dim * 2)
        mask: 注意力掩码，形状为 (seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        """
        orig_dtype = x.dtype
        attn_output = self.attention(self.norm1(x), freqs_cis, mask)
        x = (x.float() + attn_output.float()).to(orig_dtype)

        # 前馈网络
        ffn_output = self.ffn(self.norm2(x))
        x = (x.float() + ffn_output.float()).to(orig_dtype)

        return x


def main():
    # 测试注意力模块
    print("\nTesting Attention:")
    attention = Attention(hidden_size=12, heads=4, groups=2)
    x = torch.randn(3, 5, 12)
    print(f"Input shape: {x.shape}")
    output = attention(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
