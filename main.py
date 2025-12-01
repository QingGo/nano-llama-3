
'''
用 PyTorch 实现 `RMSNorm` 类和 `SwiGLU` 模块
'''

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
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
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


if __name__ == "__main__":
    main()
