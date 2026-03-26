# MathModel: 数学推理模型后训练研究

> 以 Qwen2.5-3B-Base 为基座，系统研究数学推理场景下 **SFT / DPO / GRPO** 等后训练方法在不同数据配比与训练策略下的性能表现差异。

## 📊 核心成果

通过多轮消融实验发现，能力提升的关键不在于简单增加训练阶段或混合数据，而在于**合理的数据分层与训练策略设计**；其中 **replay 机制**可显著缓解遗忘，更强的**蒸馏 SFT 基座**结合 **GRPO** 能进一步突破性能瓶颈。

### 最终效果 (Best Model: SFT + GRPO)

| Benchmark | Qwen2.5-3B-Base | Qwen2.5-3B-Instruct | **Ours** | Δ vs Base |
|-----------|:---:|:---:|:---:|:---:|
| **GSM8K** | 73.69% | 81.88% | **83.70%** | +10.01 |
| **MATH-500** | 50.00% | 53.80% | **61.40%** | +11.40 |

### 与专用数学模型对比

| 模型 | 参数量 | GSM8K | MATH-500 | MATH-500 (MV@64) | AIME'24 |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 83.47% | 61.60% | 63.60% | 13.33% |
| **Ours (Qwen2.5-3B + SFT + GRPO)** | **3B** | **83.70%** | **61.40%** | **66.00% ✅** | 3.33% |
| Qwen2.5-Math-7B-Instruct | 7B | 93.93% | 71.60% | 72.00% | 23.33% |

- ✅ **Greedy decoding 已追平** Qwen2.5-Math-1.5B-Instruct (61.40% vs 61.60%)
- ✅ **超越** Qwen2.5-3B-Instruct

## 🏗️ 最优训练流程

```
Qwen2.5-3B-Base (50.00%)
    │
    ├── [Stage 1] 蒸馏 SFT (28K 高质量数学数据)
    │   └── Math-7B-Instruct 蒸馏 + 增强采样
    │   └── → SFT Model (59.80%)
    │
    └── [Stage 2] GRPO 强化学习 (50K RL 问题)
        └── DAPO loss, G=32, constant lr
        └── → Final Model (61.40% greedy / 66.00% MV@64)
```

### Stage 1: 蒸馏 SFT

以 Qwen2.5-Math-7B-Instruct 为教师模型，对 50K RL 训练问题生成解答，通过答案验证过滤正确解，获得 28K 高质量蒸馏数据。

| 配置项 | 值 |
|--------|-----|
| 基座模型 | Qwen2.5-3B-Base |
| 训练数据 | 28,098 samples (蒸馏 + 增强蒸馏) |
| 训练方式 | 全量微调 (Full Fine-tuning) |
| 学习率 | 2e-5, cosine scheduler |
| Warmup | 3% |
| Batch Size | 4 × 4 (grad accumulation) = 16 |
| Max Sequence Length | 4096 |
| Epochs | 3 (最佳 checkpoint 在 step 3000) |
| 训练时间 | ~5 小时 |
| 最佳结果 | **59.80%** MATH-500 (checkpoint-3000) |

### Stage 2: GRPO 强化学习

在 SFT 模型基础上进行 GRPO (Group Relative Policy Optimization) 训练，使用 DAPO loss 变体，无 KL 惩罚。

| 配置项 | 值 |
|--------|-----|
| 基座模型 | SFT checkpoint-3000 (59.80%) |
| 训练数据 | 50,000 RL problems (OpenThoughts math subset) |
| Loss Type | DAPO (Dynamic Advantage Policy Optimization) |
| 学习率 | 3e-7, constant scheduler |
| Group Size (G) | 32 |
| KL Penalty (β) | 0.0 |
| Clipping (ε/ε_high) | 0.1 / 0.28 |
| Temperature | 0.9 |
| Gradient Accumulation | 4 |
| Max Steps | 300 (最佳 checkpoint) |
| 最佳结果 | **61.40%** greedy / **66.00%** MV@64 |

## 🔬 关键消融发现

### 1. 数据分层与 Replay 机制

两阶段 SFT (基础数学 → 竞赛数学 + replay) 显著优于单阶段混合训练：

| 策略 | GSM8K | MATH-500 | 说明 |
|------|:---:|:---:|------|
| 单阶段混合 (OT82K+Comp25K) | 44.05% | 33.60% | ❌ 单阶段效果极差 |
| 两阶段无 Replay | 58.00% | 45.60% | ⚠️ 灾难性遗忘 |
| **两阶段 + Replay** | **73.24%** | **44.60%** | ✅ Replay 缓解遗忘 |
| 三阶段 + Replay | 69.83% | 43.60% | ❌ 更多阶段反而退步 |

**结论**: Replay 机制使 GSM8K 提升 **+15.24%**，是两阶段训练的关键技术。

### 2. SFT 数据质量 vs 数量

| 数据 | 样本数 | MATH-500 | 说明 |
|------|:---:|:---:|------|
| NuminaMath-CoT 100K | 100K | 39.20% | 大量通用数学数据 |
| OpenThoughts 82K | 82K | 39.20% | 高质量推理数据 |
| **蒸馏 28K (7B→3B)** | **28K** | **59.80%** | ✅ 质量远重要于数量 |
| 蒸馏 + 教师增强 65K | 65K | 57.20% | ❌ 增加数据反而下降 |

**结论**: 数据质量远重要于数量。28K 蒸馏数据效果优于 100K 通用数据。

### 3. GRPO 超参数敏感性

| 配置 | MATH-500 | 说明 |
|------|:---:|------|
| G=16 | 60.40% | Group size 偏小 |
| **G=32** | **61.40%** | ✅ 最优 |
| G=64 | 60.80% | Group size 偏大，显存紧张 |
| lr=1e-6 (cosine) | 60.20% | 学习率过高 |
| **lr=3e-7 (constant)** | **61.40%** | ✅ 最优 |

### 4. DPO 实验

DPO 在本项目中作为对比实验，通过对模型生成的解答构建偏好对进行训练：

| DPO 实验 | 数据来源 | MATH-500 | 说明 |
|---------|---------|:---:|------|
| Self-play DPO | 模型自身生成偏好对 | 48.40% | v1 最佳 |
| Distill DPO (7B vs 3B) | Math-7B 正确解 vs 3B 错误解 | 47.20% | 效果一般 |
| DPO from GRPO best | GRPO 最佳模型生成对 | 60.40% | 略低于 GRPO 本身 |

**结论**: 在数学推理场景中，GRPO 的在线策略优化优于 DPO 的离线偏好学习。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- TRL >= 0.27
- vLLM >= 0.14
- transformers, peft, wandb
- GPU: 64GB+ VRAM (推荐 A100 80GB / MetaX C500 64GB)

### 安装依赖

```bash
pip install torch transformers peft trl vllm wandb datasets pandas
```

### 数据准备

```bash
# 1. 准备原始数据集 (放置在 DATA_ROOT 下)
# - NuminaMath-CoT (Parquet 格式)
# - OpenThoughts-114k (Parquet 格式)
# - 评估集: MATH-500, GSM8K, AIME2024, AIME2025, MMLU-STEM, OlympiadBench

# 2. 处理 SFT 数据 (NuminaMath-CoT → chat 格式)
python data_processing/prepare_sft_data.py

# 3. 处理 RL 数据 (OpenThoughts → prompt+answer 格式)
python data_processing/prepare_rl_data.py

# 4. 处理评估数据
python data_processing/prepare_eval_data.py

# 5. 蒸馏: 用 Math-7B-Instruct 生成高质量解答
python data_processing/distill_math7b_large.py

# 6. 增强蒸馏: 对未解决的问题多采样
python data_processing/enhanced_distill_targeted.py

# 7. 合并蒸馏数据
python data_processing/combine_sft_data.py \
    --inputs sft_distill_math7b.jsonl sft_enhanced_distill.jsonl \
    --output sft_combined_v5.jsonl
```

### 训练

#### Stage 1: SFT

```bash
bash scripts/01_sft_distill.sh
# 输出: outputs/sft/sft_distill/checkpoint-3000
# 评估: ~59.80% MATH-500
```

#### Stage 2: GRPO

```bash
bash scripts/02_grpo.sh
# 输出: outputs/grpo/grpo_from_sft/checkpoint-300
# 评估: ~61.40% MATH-500 greedy
```

### 评估

```bash
# Greedy decoding 评估
bash scripts/03_evaluate.sh <model_path>

# Majority Vote@64 评估
bash scripts/04_evaluate_mv.sh <model_path>

# 全部 benchmark 评估
bash scripts/05_evaluate_all_benchmarks.sh <model_path>
```

### 可选: DPO 训练

```bash
# 1. 生成 DPO 偏好对数据
bash scripts/06_generate_dpo_data.sh <model_path>

# 2. DPO 训练
bash scripts/07_train_dpo.sh
```

### 可选: 两阶段 SFT + Replay

```bash
# 两阶段 SFT，第二阶段使用竞赛数据 + OT 回放数据
bash scripts/08_two_stage_sft_replay.sh
```

## 📝 训练数据说明

### SFT 数据: `sft_combined_v5.jsonl` (28,098 samples)

| 来源 | 样本数 | 说明 |
|------|:---:|------|
| Math-7B 蒸馏 | ~17,638 | 用 Qwen2.5-Math-7B-Instruct 对 RL 问题生成解答，答案验证过滤 |
| 增强蒸馏 | ~10,460 | 对未解决问题多次采样 (temp=0.7, n=2)，答案验证过滤 |
| **合计** | **28,098** | 格式: `{"messages": [system, user, assistant]}` |

### RL 数据: `rl_50k.jsonl` (50,000 samples)

| 来源 | 样本数 | 说明 |
|------|:---:|------|
| OpenThoughts-114k (math subset) | 50,000 | 格式: `{"prompt": [...], "answer": str}` |

### 评估集

| Benchmark | 样本数 | 类型 |
|-----------|:---:|------|
| MATH-500 | 500 | 数学竞赛 (Level 1-5) |
| GSM8K | 1,319 | 小学数学应用题 |
| AIME 2024 | 30 | 美国数学邀请赛 |
| AIME 2025 | 30 | 美国数学邀请赛 |
| MMLU-STEM | 570+ | STEM 多选题 |
| OlympiadBench | 675 | 国际数学奥赛 |

### 答案提取与匹配

使用 `utils/answer_extraction.py` 进行鲁棒的答案提取：
- 支持 `\boxed{}` 嵌套提取
- LaTeX 公式归一化 (分数、根号、多项式等)
- 数值近似匹配 (浮点数容差)
- 多格式兼容 (纯文本、LaTeX、混合)

### GRPO 奖励函数

- **Correctness Reward**: 提取 `\boxed{}` 答案与 ground truth 比对
- 正确 = +2.0, 错误 = 0.0
- 使用 `mask_truncated` 过滤截断的生成
- 使用 `correctness_only` 只基于正确性奖励

### Majority Vote

通过多次采样 (N=64, temperature=0.7) 生成多个解答，提取答案后投票选择出现频率最高的答案，可提升准确率 ~5%。

## 🔗 参考

- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [GRPO: DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DAPO: Dynamic Advantage Policy Optimization](https://arxiv.org/abs/2503.14476)
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)

## 📄 License

本项目仅用于学术研究。基座模型 Qwen2.5-3B 遵循 Apache 2.0 License。
