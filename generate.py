import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from llama import Llama, load_safetensors_weights


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def load_llama3_tokenizer(model_path):
    """
    使用 tiktoken 加载 Llama 3 的 tokenizer.model
    """
    # 1. 加载基础 BPE 数据 (mergeable ranks)
    # tiktoken 会自动处理文件中的 base64 编码
    mergeable_ranks = load_tiktoken_bpe(model_path)

    # 2. 定义 Llama 3 的特殊 Token
    # 官方定义的特殊 token 列表
    special_tokens_list = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # End of Turn
    ]

    # Llama 3 预留了 256 个特殊 token 插槽
    # 剩下的填充为 reserved_special_token_5 ... 255
    num_reserved_special_tokens = 256
    for i in range(len(special_tokens_list), num_reserved_special_tokens):
        special_tokens_list.append(f"<|reserved_special_token_{i}|>")

    # 构建 special_tokens 字典: {token_str: token_id}
    # 特殊 token 的 ID 是紧接在普通词表之后的
    num_base_tokens = len(mergeable_ranks)
    special_tokens = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens_list)
    }

    # 3. 构造 tiktoken 对象
    # pat_str 是官方使用的正则表达式模式，用于切分文本
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py 里 cl100k_base 的 pat_str 应该是：
    # r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
    enc = tiktoken.Encoding(
        name="llama3",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    return enc


if __name__ == "__main__":
    # 可以用 modelscope 单独下载词表文件，用于本地测试
    # modelscope download --model LLM-Research/Meta-Llama-3-8B special_tokens_map.json tokenizer.json tokenizer_config.json original/tokenizer.model --local_dir ./
    # tt_model_file = "./tokenizer.model"
    # model_path = "./"
    model_path = "./llama3-8B"    
    tt_model_file = model_path + "/original/tokenizer.model"
    tt_tokenizer = load_llama3_tokenizer(tt_model_file)
    print(f"tiktoken 词表大小: {tt_tokenizer.n_vocab}")  # 应该是 128256
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # 在 HF 中，vocab_size 属性通常指的是基础词表大小，是 128000
    print(f"Hugging Face tokenizer 词表大小: {hf_tokenizer.vocab_size}")

    # 测试编码
    text = "The capital of France is"
    # Hugging Face 会默认会在句子开头添加 <|begin_of_text|>（ID: 128000），
    # 所以需要在 tiktoken 中额外对齐这个行为
    tt_tokens = tt_tokenizer.encode(
        "<|begin_of_text|>" + text, allowed_special={"<|begin_of_text|>"}
    )
    hf_tokens = hf_tokenizer.encode(text)
    print(f"'{text}' -> tiktoken IDs: {tt_tokens}")
    print(f"'{text}' -> HuggingFace IDs: {hf_tokens}")
    assert tt_tokens == hf_tokens, "tiktoken 和 Hugging Face tokenizer 编码结果不一致"

    # 使用自定义的 Llama 模型进行预测
    device = get_device()
    print(f"Using device: {device}")
    custom_model = Llama(vocab_size=128256, hidden_size=4096, ffn_hidden_size=14336, heads=32, groups=8, dtype=torch.bfloat16)
    custom_model.to(device)
    index_path = model_path + "/model.safetensors.index.json"
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
    load_safetensors_weights(
        custom_model,
        model_path,
        weight_map,
        device,
        True,
    )
    custom_model.eval()
    # 传入 tt_tokens
    with torch.no_grad():
        tt_output = custom_model(torch.tensor([tt_tokens], device=device))
        print(tt_output)
    # 从 内存卸载 custom_model
    del custom_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 使用 Hugging Face 模型进行预测
    # 需要指定 torch_dtype，默认行为通常是加载为 float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    )
    hf_model.to(device)
    with torch.no_grad():
        hf_output_obj = hf_model(torch.tensor([hf_tokens], device=device))
        hf_output = hf_output_obj.logits
        print(hf_output)

    # 对比 tt_output 和 hf_output
    assert tt_output.shape == hf_output.shape, (
        "自定义模型和 Hugging Face 模型输出形状不一致"
    )
    assert torch.allclose(tt_output, hf_output, atol=1e-5), (
        "自定义模型和 Hugging Face 模型输出数值不一致"
    )

    # 解码
    tt_decoded = tt_tokenizer.decode(tt_tokens)
    hf_decoded = hf_tokenizer.decode(hf_tokens)
    print(f"'{tt_decoded}' -> tiktoken 解码结果")
    print(f"'{hf_decoded}' -> HuggingFace 解码结果")
