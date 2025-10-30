# 🧠 BASS++：面向大语言模型的高吞吐推测推理算法与系统优化

*(Batched Attention-optimized Speculative Sampling for Efficient Large Language Model Inference)*

------

## 一、项目简介

本项目基于 **推测推理（Speculative Decoding）** 技术，提出并实现了 **BASS++（Batched Attention-optimized Speculative Sampling）** 框架，
 用于在保持生成质量的前提下显著提升大语言模型（LLM）的推理速度与并行吞吐性能。

本仓库包含：

- **任务一**：推测推理与批量加速算法实现（Speculative / BASS）；
- **任务二**：基于 Dolly 数据集与自定义批次的性能评测与分析。

------

## 二、项目结构

```
LLMSpeculativeSampling/
├── main.py                  # 主推理脚本（任务一）
├── eval_dolly.py            # 基于 Dolly 数据集的评测脚本（任务二）
├── serving.py               # Flask Web 服务接口（可选）
├── requirements.txt         # 环境依赖
│
├── benchmark/               # 任务二批量测试与分析模块
│   ├── analyzer.py
│   ├── metrics.py
│   ├── run_experiment.py
│   ├── runner.py
│   ├── runner_support.py
│   └── utils_io.py
│
├── sampling/                # 核心算法模块
│   ├── __init__.py
│   ├── autoregressive_sampling.py
│   ├── speculative_sampling.py
│   ├── speculative_bass.py
│   ├── kvcache_model.py
│   ├── batched_kvcache_model.py
│   ├── utils.py
│   ├── utils_bass.py
│   └── globals.py
│
├── results/                 # 实验输出目录（默认保存在本地根目录）
└── LICENSE.txt
```

------

## 三、数据与模型目录说明

> ⚠️ 注意：`data/` 目录仅存在于服务器端，本地开发环境中无需同步。

服务器端目录结构示例：

```
data/
├── dataset/
│   └── dolly/
│       └── databricks-dolly-15k.jsonl
└── models/
    ├── bloom-560m/
    ├── bloomz-7b1/
    ├── TinyLlama-1B/
    └── Llama-2-7b-raw/
```

本地运行脚本时通过相对路径 `../data/models/...` 引用。

------

## 四、算法原理简述

### 🔹 推测推理（Speculative Decoding）

- 小模型草稿预测 γ 个候选 token；
- 大模型验证接受前缀；
- 一致部分批量接受，从而减少目标模型前向调用。

### 🔹 BASS-PAD（批量推理）

- 多序列并行草稿验证；
- 使用 PAD 对齐序列；
- KV Cache 批量管理与回滚；
- 兼容 LLaMA、BLOOM、OPT 等主流架构。

### 🔹 动态草稿长度 γ 调度

根据接受率 α 自适应更新草稿长度：
 [
 \gamma_{t+1} =
 \begin{cases}
 \min(\gamma_t + \Delta_1, \gamma_{\max}), & \text{若全部接受}[4pt]
 \max(1, \max(x_i), \gamma_t - \lceil \gamma_t/m \rceil - s), & \text{否则}
 \end{cases}
 ]

------

## 五、运行环境配置

### 1️⃣ 创建环境

```bash
conda create -n bass python=3.10
conda activate bass
pip install -r requirements.txt
```

### 2️⃣ 推荐环境变量

在 PyCharm 或命令行运行时建议设置：

```bash
PYTHONUNBUFFERED=1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

------

## 六、脚本使用方法

### 🔹 1. main.py —— 单条或批量推理（任务一）

命令行示例（BLOOM 模型）：

```bash
python main.py \
  --input "The quick brown fox jumps over the lazy " \
  --target_model_name ../data/models/bloomz-7b1 \
  --approx_model_name ../data/models/bloom-560m \
  -v --bass --batch 32
```

也可将模型替换为 LLaMA：

```bash
python main.py \
  --input "人工智能的推理机制包括哪些方面？" \
  --target_model_name ../data/models/Llama-2-7b-raw \
  --approx_model_name ../data/models/TinyLlama-1B \
  --bass --batch 16
```

**参数说明：**

| 参数                  | 类型 | 说明                       |
| --------------------- | ---- | -------------------------- |
| `--input`             | str  | 输入文本前缀               |
| `--target_model_name` | path | 目标模型路径               |
| `--approx_model_name` | path | 草稿模型路径               |
| `--batch`             | int  | 批大小                     |
| `--bass`              | flag | 启用 BASS-PAD 批量推理模式 |
| `-v`                  | flag | 输出详细日志               |

------

### 🔹 2. eval_dolly.py —— Dolly 数据集批量评测（任务二）

命令示例：

```bash
python eval_dolly.py \
  --target_model_name ../data/models/Llama-2-7b-raw \
  --approx_model_name ../data/models/TinyLlama-1B \
  --batch 32 \
  --samples 256 \
  --max_tokens 20 \
  --gamma_init 7 \
  --temperature 0.9 \
  --top_p 0.9 \
  --top_k 100 \
  --modes deepmind,google,BASS \
  --dataset_path ../data/dataset/dolly/databricks-dolly-15k.jsonl
```

该脚本会运行三种模式（DeepMind / Google / BASS），在 Dolly 数据集上评估推理时间与生成结果一致性。

> ⚠️ 注意：本脚本 **不会自动写入 results 目录**，而是直接在控制台输出性能统计。

**主要参数：**

| 参数                  | 类型  | 说明                   |
| --------------------- | ----- | ---------------------- |
| `--target_model_name` | path  | 目标模型路径           |
| `--approx_model_name` | path  | 草稿模型路径           |
| `--batch`             | int   | 批大小                 |
| `--samples`           | int   | 采样样本数             |
| `--max_tokens`        | int   | 每条生成最大 token 数  |
| `--gamma_init`        | int   | 草稿初始步长           |
| `--temperature`       | float | softmax 温度           |
| `--top_p`             | float | nucleus sampling 参数  |
| `--top_k`             | int   | top-k 采样参数         |
| `--modes`             | list  | 指定实验模式（可多选） |
| `--dataset_path`      | path  | Dolly 数据集路径       |

------

### 🔹 3. benchmark.run_experiment —— 模块化任务二性能分析

`benchmark/run_experiment.py` 为模块执行脚本，推荐以 *module* 方式运行（PyCharm 或命令行均可）：

命令行示例：

```bash
python -m benchmark.run_experiment
```

或在 PyCharm “Run Configuration” 中配置：

- **Module name:** `benchmark.run_experiment`
- **Working directory:** 项目根目录 `LLMSpeculativeSampling`
- **Environment variables:**
   `PYTHONUNBUFFERED=1;PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`

该模块负责任务二的完整性能测试流程，包括：

- 加载模型组合；
- 自动批量推理；
- 记录平均吞吐与延迟；
- 生成 CSV 与图表输出（存放在 `results/` 下）。

------

## 七、实验结果示例

| 模型组合                | 模式                                | 吞吐提升 | 延迟降低  | GPU 利用率 |
| ----------------------- | ----------------------------------- | -------- | --------- | ---------- |
| Bloom-560M + Bloomz-7B1 | Baseline                            | 1×       | 1×        | 4%         |
| ↑                       | Speculative Decoding                | 2.1×     | 0.65×     | 8%         |
| ↑                       | BASS-PAD                            | **2.8×** | **0.55×** | **12%**    |
| ↑                       | BASS++ (γ 自适应 + Sorted Batching) | **3.1×** | **0.45×** | **15%+**   |

------

## 八、许可证与引用

本项目基于 [FeiFeiBear/speculative-decoding-demo](https://github.com/FeiFeiBear/speculative-decoding-demo)
 开源实现进行二次开发，遵循 **Apache License 2.0**。

引用论文：

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv 2023.
- AWS AI Lab, *BASS: Speculative Sampling for Batch Decoding*, 2024.

------

## 九、联系方式

- 团队名称：**DeepSpec 深推团队**
- 联系人：XXX
- 邮箱：[xxx@xxx.edu.cn](mailto:xxx@xxx.edu.cn)
- 日期：2025 年 10 月

------

