"""
用 PyTorch 实现 `RMSNorm` 类和 `SwiGLU` 模块
"""

from typing import Optional, List
import torch
import torch.nn as nn
from torchview import draw_graph
from safetensors.torch import safe_open
import json
import re

from util import RMSNorm
from transformer import TransformerBlock


class Llama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        ffn_hidden_size: int,
        heads: int,
        groups: Optional[int] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        实现Llama模型，包含多个Transformer块
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度
            ffn_hidden_size: 前馈网络隐藏层维度
            heads: 头数
            groups: 分组数
            dropout: dropout概率
            dtype: 参数数据类型，默认torch.bfloat16
        """
        super().__init__()
        self.dtype = dtype
        self.embedding = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, ffn_hidden_size, heads, groups, dropout, dtype=dtype)
                for _ in range(32)  # llama3-8B 有 32 个Transformer块
            ]
        )
        self.norm = RMSNorm(hidden_size, dtype=dtype)
        # 映射回词汇表大小，不需要 bias
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype)

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

        # 输出层，映射回词汇表 logits
        x = self.lm_head(x)

        return x


params_map = {
    "embedding.weight": "model.embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "transformer_blocks.0.attention.w_k.weight": "model.layers.0.self_attn.k_proj.weight",
    "transformer_blocks.0.attention.w_o.weight": "model.layers.0.self_attn.o_proj.weight",
    "transformer_blocks.0.attention.w_q.weight": "model.layers.0.self_attn.q_proj.weight",
    "transformer_blocks.0.attention.w_v.weight": "model.layers.0.self_attn.v_proj.weight",
    "transformer_blocks.0.ffn.swiglu.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
    "transformer_blocks.0.ffn.swiglu.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
    "transformer_blocks.0.ffn.swiglu.up_proj.weight": "model.layers.0.mlp.up_proj.weight",
    "transformer_blocks.0.norm1.weight": "model.layers.0.input_layernorm.weight",
    "transformer_blocks.0.norm2.weight": "model.layers.0.post_attention_layernorm.weight",
}


def map_weight_name(weight_name: str) -> str:
    """
    将 safetensors 中的权重名映射到自定义模型中的权重名
    Args:
        weight_name: safetensors 中的权重名
    Returns:
        自定义模型中的权重名
    """
    # 提取层索引, e.g. transformer_blocks.0.attention.w_q.weight -> 0
    layer_idx = None
    if "transformer_blocks" in weight_name:
        layer_idx = re.search(r"transformer_blocks\.(\d+)\.", weight_name).group(1)
        # 替换层索引为 0
        weight_name = weight_name.replace(
            f"transformer_blocks.{layer_idx}", "transformer_blocks.0"
        )

    # 映射权重名
    official_weight_name = params_map.get(weight_name, weight_name)
    if layer_idx is not None:
        official_weight_name = official_weight_name.replace(
            ".layers.0", f".layers.{layer_idx}"
        )
    return official_weight_name


def inspect_structure(
    weight_map: dict[str, str], model_path: str, model: Llama
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """
    检查模型结构是否与 safetensors 文件中的权重匹配，
    并将模型结构和 safetensors 文件中的权重结构保存到 JSON 文件中。
    后续可以用来开发权重名称的映射函数
    Args:
        weight_map: 权重名映射字典
        model_path: 模型路径
        model: Llama 模型实例
    """
    official_shapes = {}
    safetensors_files = sorted(list(set(weight_map.values())))
    for safetensors_file in safetensors_files:
        safetensors_path = model_path + "/" + safetensors_file
        # safe_open 只是打开文件句柄，不会把几GB的数据读入内存
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # get_slice(key).get_shape() 只读取形状元数据
                shape = f.get_slice(key).get_shape()
                official_shapes[key] = list(shape)
    with open("official_structure.json", "w") as f:
        # 关键点：添加 sort_keys=True
        json.dump(official_shapes, f, indent=4, sort_keys=True)
    custom_shapes = {}
    for name, param in model.named_parameters():
        custom_shapes[name] = list(param.shape)
    with open("custom_structure.json", "w") as f:
        # 关键点：添加 sort_keys=True
        json.dump(custom_shapes, f, indent=4, sort_keys=True)
    return custom_shapes, official_shapes


def load_safetensors_weights(
    model: Llama,
    model_path: str,
    weight_map,
    device: str,
    load_all: bool = False,
    layers_to_load: Optional[List[str]] = None,
) -> None:
    """
    加载 safetensors 权重到 Llama 模型

    Args:
        model: Llama 模型实例
        model_path: 模型文件夹路径
        weight_map: 权重名映射字典
        device: 加载权重的设备
        load_all: 是否加载所有权重，默认为 False
        layers_to_load: 如果 load_all 为 False，指定要加载的层名称列表
    """

    model_params = dict(model.named_parameters())

    if load_all:
        layers_to_load = list(model_params.keys())

    for layer_name in layers_to_load:
        official_name = map_weight_name(layer_name)
        # 需要找到权重文件
        safetensors_file = weight_map[official_name]
        safetensors_path = model_path + "/" + safetensors_file
        # safe_open 只是打开文件句柄，不会把几GB的数据读入内存
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            params = f.get_tensor(official_name)
            # 更新模型状态字典中的权重
            with torch.no_grad():
                model_params[layer_name].copy_(params.to(device))
            print(f"Loaded {official_name} from {safetensors_path}")

def weight_stats(weight: torch.Tensor) -> None:
    """
    打印权重的统计信息
    Args:
        weight: 权重张量
    """
    print(f"  权重形状: {weight.shape}")
    print(f"  均值: {weight.mean().item():.6f}")
    print(f"  方差: {weight.var().item():.6f}")
    print(f"  最大值: {weight.max().item():.6f}")
    print(f"  最小值: {weight.min().item():.6f}")
    print(f"  非零元素比例: {(weight != 0).float().mean().item():.6f}")

def test_llama():
    # meta 模式测试，不加载实际权重
    with torch.device("meta"):
        # 创建模型，参数参考 llama3 技术报告 Table 3
        model = Llama(
            vocab_size=128256,
            hidden_size=4096,
            ffn_hidden_size=14336,
            heads=32,
            groups=8,
        )
    meta_input_ids = torch.randint(0, 10000, (1, 128), device="meta")
    model_graph = draw_graph(
        model,
        input_data=meta_input_ids,
        expand_nested=True,
        depth=2,  # 2/3
        device="meta",
    )
    filename = "llama3_structure_meta"
    model_graph.visual_graph.render(filename, format="png")
    model_path = "/Volumes/My Passport/model/Llama-3-8B"
    index_path = model_path + "/model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
    custom_shapes, official_shapes = inspect_structure(weight_map, model_path, model)
    # 验证 custom_shapes 每个都能映射到 official_shapes
    print(
        f"custom_shapes 大小，{len(custom_shapes)},  official_shapes 大小，{len(official_shapes)}"
    )
    for custom_name, custom_shape in custom_shapes.items():
        try:
            official_name = map_weight_name(custom_name)
            official_shape = official_shapes[official_name]
            assert custom_shape == official_shape, (
                f"custom_shape {custom_shape} != official_shape {official_shape}"
            )
        except KeyError as e:
            print(f"Warning: {custom_name} -> {official_name} not in weight map")
            raise e
    print("所有 custom_shapes 都能映射到 official_shapes")
    
    # 将模型从 meta 设备转移到 cpu，使用 to_empty() 初始化空张量。不能使用 model.to()
    model = model.to_empty(device="cpu")
    
    # 测试加载 transformer_blocks.0.attention.w_q.weight 权重
    test_layer_name = "transformer_blocks.0.attention.w_q.weight"
    
    # 加载前检查权重
    print(f"\n加载前 - {test_layer_name}:")
    pre_weight = model.state_dict()[test_layer_name]
    weight_stats(pre_weight)
    
    # 加载权重
    load_safetensors_weights(
        model,
        model_path,
        weight_map,
        "cpu",
        False,
        [test_layer_name],
    )
    
    # 加载后检查权重
    print(f"\n加载后 - {test_layer_name}:")
    post_weight = model.state_dict()[test_layer_name]
    weight_stats(post_weight)


def main():
    # 测试 llama 模型
    print("\nTesting Llama Model:")
    test_llama()


if __name__ == "__main__":
    main()
