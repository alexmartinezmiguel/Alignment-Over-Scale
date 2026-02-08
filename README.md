# Alignment Over Scale: LLM-Based News Recommendation via Reinforcement Fine-Tuning

This repository contains the code for fine-tuning large language models (LLMs) as personalized news recommenders on the [MIND dataset](https://msnews.github.io/). We compare **Supervised Fine-Tuning (SFT)** with **Group Relative Policy Optimization (GRPO)**, a reinforcement-learning approach that aligns model outputs using multiple reward signals — recommendation quality (nDCG), output format validity, and reasoning depth.

The codebase supports three model families out of the box: **Gemma**, **Llama**, and **Qwen**, all fine-tuned efficiently via **LoRA** adapters.

---

## Repository Structure

```
Alignment-Over-Scale/
├── finetune_llm_sft.py            # SFT training script
├── finetune_llm_grpo.py           # GRPO training script
├── llm_recommender_ndcg_analysis.py  # Checkpoint evaluation (nDCG, AUC, MRR)
├── utils/
│   ├── data_utils.py              # Dataset construction (chat templates, SFT pairs, ROUGE)
│   ├── training_utils.py          # Reward functions & ranking quality analysis
│   ├── experiment_manager.py      # Experiment directory & config management
│   └── misc_utils.py              # Logging, seeding, argument helpers
├── data/
│   └── mind/
│       └── MIND_test.csv          # MIND test split (sample)
├── output/                        # Training outputs (checkpoints, metrics, configs)
├── Dockerfile                     # GPU-enabled Docker environment
├── requirements.txt               # Base dependencies (TensorFlow, recommenders)
└── llm_requirements.txt           # LLM-specific dependencies (transformers, trl, peft)
```

## Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker (recommended) or a local Python 3.10+ environment
- A [Hugging Face](https://huggingface.co/) account and API token
- A [Weights & Biases](https://wandb.ai/) account and API key

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Alignment-Over-Scale.git
cd Alignment-Over-Scale
```

### 2. Configure API tokens

Open each training script and replace the placeholder tokens with your own credentials:

```python
HF_TOKEN = 'YOUR_HF_TOKEN_HERE'        # Your Hugging Face token
WANDB_API_KEY = 'YOUR_WANDB_API_KEY_HERE'  # Your Weights & Biases API key
```

These appear in:
- `finetune_llm_sft.py`
- `finetune_llm_grpo.py`
- `llm_recommender_ndcg_analysis.py` (HF token only)

### 3a. Docker (recommended)

```bash
# Build the image
docker build -t alignment-over-scale .

# Run with GPU access, mounting the repo into /workspace
docker run --gpus all -it \
  -v $(pwd):/workspace \
  alignment-over-scale
```

### 3b. Local installation

```bash
pip install -r requirements.txt
pip install --no-deps recommenders[gpu]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r llm_requirements.txt
```

### 4. Prepare the data

Place the processed MIND dataset files in `data/mind/processed_recommender/`.


For GRPO training with the nDCG reward, you also need pre-computed baseline recommender scores:

```
output/recommender/train/<recommender_name>/ranking_metrics.pkl
```

This pickle file maps impression IDs to their baseline metric scores (e.g., from LSTUR).

## Usage

### Supervised Fine-Tuning (SFT)

```bash
python finetune_llm_sft.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --prompt_version v8 \
    --num_train_epochs 20 \
    --learning_rate 1.41e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 12
```

### GRPO Fine-Tuning

```bash
python finetune_llm_grpo.py \
    --model_name google/gemma-2-2b-it \
    --prompt_version v7 \
    --reward_type v5 \
    --recommender_metric ndcg@10 \
    --recommender_rewards_file lstur_recprompt_mind_e5_bs256 \
    --num_train_epochs 20 \
    --num_generations 12 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 12
```


### Checkpoint Evaluation

Evaluate a range of training checkpoints on the test set:

```bash
python llm_recommender_ndcg_analysis.py \
    --model-type google/gemma-2b-it \
    --batch-size 4
```

This iterates over specified checkpoints, generates ranked recommendations for each test impression, and saves metrics (nDCG@5, nDCG@10, AUC, MRR) as CSV files to the output directory.

## Supported Models

| Family | Example Model | LoRA Config |
|--------|--------------|-------------|
| Gemma | `google/gemma-2-2b-it` | r=64, α=16 |
| Llama | `meta-llama/Llama-3.2-3B-Instruct` | r=32, α=64 |
| Qwen | `Qwen/Qwen2.5-3B-Instruct` | r=8, α=32 |

All models are fine-tuned with LoRA applied to the `q_proj` and `v_proj` attention layers.

## Key Dependencies

| Package | Version |
|---------|---------|
| PyTorch | CUDA 11.8 |
| Transformers | 4.55.0 |
| TRL | 0.21.0 |
| PEFT | latest |
| TensorFlow | 2.12.0 |
| Microsoft Recommenders | latest (GPU) |

See `requirements.txt` and `llm_requirements.txt` for the full list.

