# nano-llama-3

一个以学习为目标的「Llama 3 架构复现」项目：不依赖 `transformers` 高层实现，只用 PyTorch 原生算子搭建核心模块（RMSNorm、RoPE、GQA、SwiGLU），加载官方权重并与 HuggingFace 实现进行数值对齐。

# 相关知乎文章
* 总计划：[五年INTP失业程序员，半年计划勇闯大模型岗位](https://zhuanlan.zhihu.com/p/1978508668996718986)
* Week 1-2 日计划：[【每日计划】Week 1-2 大模型学习](https://zhuanlan.zhihu.com/p/1978509667509482382)
* [【学习笔记】week1.1 环境搭建与基础组件 (RMSNorm & SwiGLU)](https://zhuanlan.zhihu.com/p/1978931693940392735)
* [【学习笔记】week1.2 旋转位置编码 (RoPE)](https://zhuanlan.zhihu.com/p/1979325979487933459)
* [【学习笔记】week1.3 GQA (分组查询注意力)](https://zhuanlan.zhihu.com/p/1979657319290791046)
* [【学习笔记】week1.4 llama3模型组装与权重加载](https://zhuanlan.zhihu.com/p/1980064017188926619)
* [【学习笔记】week1.5 BPE 与精度对齐](https://zhuanlan.zhihu.com/p/1980755249925997707)

在小红书也发了相关笔记。

## 项目结构

```text
.
├── generate.py                 # 推理与数值对齐脚本（HF vs 自研）
├── llama.py                    # Llama 顶层模型，权重映射与加载
├── transformer.py              # Attention / FeedForward / TransformerBlock
├── util.py                     # RMSNorm / SwiGLU / RoPE 实现与工具
├── official_structure.json     # 官方 safetensors 权重结构导出
├── custom_structure.json       # 自研模型参数结构导出
├── pyproject.toml              # 项目依赖与 Python 版本约束
├── requirements.txt            # 为服务器准备的，使用 pip 安装的依赖
├── uv.lock                     # 使用 `uv` 的锁文件
└── README.md                   # 你正在看的文件
```

## 环境与运行

- Python 版本：`>=3.13`
- 包管理器：优先使用 `uv`

安装依赖：

```bash
uv sync
```

本地运行推理与对齐（需先准备 Llama 3 权重与 tokenizer）：

```bash
uv run python generate.py
```

权重下载（示例，可选其一）：

- 使用 ModelScope CLI：`modelscope download --model LLM-Research/Meta-Llama-3-8B --local_dir <你的路径>`
- 使用 HuggingFace 离线文件：将 `model.safetensors*` 与 `tokenizer.*` 放到 `./llama3-8B` 目录（`generate.py` 默认读取）

## 学习与收获（Week 1）

本周目标与度量：

- 目标：纯 PyTorch 复现 Llama 3 核心组件（RMSNorm、RoPE、GQA、SwiGLU）。
- 对齐：同输入下 logits 与 HF 误差 < 1e-3（实测对齐到 1e-3 量级，已验证）。

### Day 1：RMSNorm 与 SwiGLU

- RMSNorm 与 LayerNorm 对比：去掉 Mean，仅保留 Variance，推导与性质梳理。
- 代码实现：`util.py` 内 `RMSNorm` 与 `SwiGLU`；均不使用 bias，更贴近 Llama 系列实现。
- 认知点：分布式视角下 RMSNorm 计算/通信略简（不需减均值）；门控激活（SwiGLU）在梯度与表达力上优于 ReLU。

### Day 2：RoPE（旋转位置编码）

- 从绝对/相对位置编码的目标函数推导出旋转变换；偶数维两两配对旋转实现相对位置信息注入。
- 实现要点：预计算 `cos/sin` 半维度频率，推理中拼接成全维；Llama 3 使用“前半/后半”配对旋转，与 HF 对齐。
- 代码位置：`util.py:95` 的 `apply_rotary_emb`，`util.py:55` 的 `precompute_freqs_cis`。

### Day 3：GQA（分组查询注意力）

- MHA/MQA/GQA 对比：推理阶段 KV Cache 是瓶颈；GQA 在保持效果的同时显著压缩 KV 开销。
- 实现要点：K/V 以分组维度计算后，使用复制广播到各头（`repeat_interleave`）。
- 代码位置：`transformer.py:88-95`（K/V 复制广播），`transformer.py:46-51`（无 bias 的投影），`transformer.py:100-113`（注意力分数与 softmax）。

### Day 4：模型组装与权重加载

- 顶层结构：`llama.py` 中 `Llama` 包含 `Embedding → N×TransformerBlock → RMSNorm → lm_head`。
- safetensors 加载：使用 `safe_open` 只读元数据/分片，`map_weight_name` 建立官方 key 到自研 key 的一一映射。
- 设备策略：`meta` + `to_empty(device)` 避免多余初始化与双份内存占用。
- 代码位置：`llama.py:17-74`（模型结构），`llama.py:153-190`（权重加载）。

### Day 5：Tokenizer 与数值对齐

- BPE 到 byte-level BPE 的演进；`tiktoken` 的正则预分词与高性能实现。
- HF 与 tiktoken 的特殊 token 行为差异：HF 自动加 `<|begin_of_text|>`，需在 tiktoken 手动对齐。
- 对齐流程：
  - 编码对齐：`generate.py:91-98`
  - 自研模型前向：`generate.py:129-148`
  - HF 模型前向：`generate.py:161-176`
  - 取下一 token 的 logits 并比较 MSE：`generate.py:183-195`

## 关键实现一览

- `RMSNorm`：`util.py:10-25`
- `SwiGLU`：`util.py:26-53`
- `RoPE` 频率与旋转：`util.py:55-93`, `util.py:95-123`
- `Attention`（含 GQA）：`transformer.py:14-123`
- `TransformerBlock`：`transformer.py:144-189`
- 顶层 `Llama`：`llama.py:17-74`
- safetensors 权重加载：`llama.py:153-190`
- 推理与 MSE 对齐：`generate.py:74-195`

## 参考资料（节选）

- Llama 3 技术报告：https://arxiv.org/abs/2407.21783
- Llama 2 技术报告：https://arxiv.org/abs/2307.09288
- RMSNorm：https://arxiv.org/abs/1910.07467
- SwiGLU：https://arxiv.org/abs/2002.05202
- RoFormer（RoPE）：https://arxiv.org/abs/2104.09864
- tiktoken：https://github.com/openai/tiktoken
- BPE（Subword Units）：https://arxiv.org/abs/1508.07909
