#!/usr/bin/env python3
"""
Group Relative Policy Optimization (GRPO) fine-tuning script for LLM-based
news recommendation on the MIND dataset.

This script fine-tunes a pre-trained language model using GRPO with multiple
reward signals: recommendation quality (nDCG), JSON format validity, and
reasoning depth. It supports Gemma, Llama, and Qwen model families via
LoRA adapters.

Usage:
    python finetune_llm_grpo.py \
        --model_name google/gemma-2-2b-it \
        --prompt_version v7 \
        --reward_type v5 \
        --recommender_metric ndcg@10 \
        --num_train_epochs 20
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import pickle
import logging
from typing import Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch # type: ignore
import pandas as pd # type: ignore
import datasets # type: ignore
from datasets import Dataset # type: ignore

from transformers import ( # type: ignore
    AutoTokenizer, # type: ignore
    HfArgumentParser, # type: ignore
    AutoModelForCausalLM, # type: ignore
    AutoConfig # type: ignore
)
from trl import GRPOConfig, GRPOTrainer # type: ignore
from peft import LoraConfig, get_peft_model # type: ignore
import evaluate # type: ignore
import wandb # type: ignore

# Set working directory for Docker container execution
os.chdir('/workspace')

# Custom utilities
from utils.misc_utils import setup_logging # type: ignore
from utils.data_utils import build_chat_template_dataset, build_rogue_dataset # type: ignore
from utils.training_utils import compute_recommendation_quality, check_json_output_format, _coerce_to_text, lenght_reasoning_content, compute_rogue_reward # type: ignore
from utils.experiment_manager import TrainingResultsManager # type: ignore

# =============================================================================
# CONSTANTS
# =============================================================================

# API tokens — replace with your own credentials
HF_TOKEN = 'YOUR_HF_TOKEN_HERE'  # Replace with your own Hugging Face token
WANDB_API_KEY = 'YOUR_WANDB_API_KEY_HERE'  # Replace with your own Weights & Biases API key

# ROUGE metric (loaded once for reuse in reward computation)
rouge = evaluate.load("rouge")

# LoRA configurations per model family
GEMMA_LORA_CONFIG = {
    'lora_alpha': 16, 'lora_dropout': 0.1, 'r': 64,
    'bias': 'none', 'task_type': 'CAUSAL_LM', 'target_modules': ['q_proj', 'v_proj']
}

LLAMA_LORA_CONFIG = {
    'r': 32, 'lora_alpha': 64, 'lora_dropout': 0.1,
    'task_type': 'CAUSAL_LM', 'target_modules': ['q_proj', 'v_proj']
}

QWEN_LORA_CONFIG = {
    'lora_alpha': 32, 'lora_dropout': 0.1, 'r': 8,
    'bias': 'none', 'task_type': 'CAUSAL_LM', 'target_modules': ['q_proj', 'v_proj']
}

# =============================================================================
# SCRIPT ARGUMENTS
# =============================================================================

@dataclass
class ScriptArguments:
    """Command-line arguments for GRPO training."""

    # Model selection
    optimization_algorithm: Optional[str] = field(default="GRPO", metadata={"help": "the optimization algorithm"})
    model_name: Optional[str] = field(default="google/gemma-2-2b-it", metadata={"help": "the model name"})

    # Training hyperparameters
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    num_train_epochs: Optional[int] = field(default=20, metadata={"help": "the number of training epochs"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": "whether to remove unused columns (needed for reward calculation)"})

    # Batching parameters
    generation_batch_size: Optional[int] = field(default=12, metadata={"help": "the generation batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=12, metadata={"help": "the number of gradient accumulation steps"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    num_generations: Optional[int] = field(default=12, metadata={"help": "the number of generations to compute the advantage"})
    torch_empty_cache_steps: Optional[int] = field(default=1, metadata={"help": "the number of steps to empty the cache"})
    num_iterations: Optional[int] = field(default=1, metadata={"help": "the number of GRPO iterations per batch"})

    # Generation parameters
    generation_kwargs: Optional[dict] = field(default=None, metadata={"help": "the generation kwargs"})
    max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the max prompt length"})
    disable_tqdm: Optional[bool] = field(default=True, metadata={"help": "whether to disable the tqdm"})

    # Logging and checkpointing
    report_to: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    log_completions: Optional[bool] = field(default=True, metadata={"help": "whether to log completions (prompt + generation)"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the logging steps"})
    output_dir: Optional[str] = field(default=os.path.join(os.getcwd(), 'output/llm_finetuning'), metadata={"help": "path to the output directory"})
    save_strategy: Optional[str] = field(default='steps', metadata={"help": "the save strategy"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the save steps"})

    # Data parameters
    data_path: Optional[str] = field(default=os.path.join(os.getcwd(), 'data/mind/processed_recommender'), metadata={"help": "path to the data"})
    cache_dir: Optional[str] = field(default='hf_cache/hub', metadata={"help": "path to cache LLM models"})
    train_dataset: Optional[str] = field(default="MIND_train.csv", metadata={"help": "the dataset name"})
    num_samples: Optional[int] = field(default=100000, metadata={"help": "the number of samples to use for training"})
    reward_type: Optional[str] = field(default="v5", metadata={"help": "the type of reward"})
    recommender_metric: Optional[str] = field(default="ndcg@10", metadata={"help": "the metric to use for the recommender"})
    recommender_rewards_file: Optional[str] = field(default="lstur_recprompt_mind_e5_bs256", metadata={"help": "the file to load the recommender rewards from"})
    prompt_version: Optional[str] = field(default="v7", metadata={"help": "the version of the prompt"})

    # Device parameters
    gpu_device: Optional[torch.device] = field(default=torch.device("cuda"), metadata={"help": "the GPU device"})
    cpu_device: Optional[torch.device] = field(default=torch.device("cpu"), metadata={"help": "the CPU device"})

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_environment(script_args: ScriptArguments) -> None:
    """Configure environment variables for HF, WandB, and CUDA memory management."""
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    os.environ['HF_HOME'] = 'hf_cache'
    os.environ['TRANSFORMERS_CACHE'] = script_args.cache_dir # type: ignore
    os.environ['HUGGINGFACE_HUB_CACHE'] = script_args.cache_dir # type: ignore
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
    datasets.disable_progress_bar()


def validate_script_args(script_args: ScriptArguments) -> None:
    """Validate that CUDA is available and required data files exist."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but required for training")

    data_file = Path(script_args.data_path) / script_args.train_dataset # type: ignore
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")


def load_data(script_args: ScriptArguments):
    """Load training data and pre-computed recommender baseline metrics.

    Args:
        script_args: Parsed command-line arguments.

    Returns:
        Tuple of (training DataFrame, recommender metrics dict).
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")

    try:
        # Load the training CSV
        data_file = Path(script_args.data_path) / script_args.train_dataset # type: ignore
        df_data = pd.read_csv(data_file)

        # Load pre-computed recommender baseline scores (used as reward reference)
        recommender_metrics_file = Path('output/recommender/train') / f'{script_args.recommender_rewards_file}' # type: ignore
        ranking_metrics_path = os.path.join(recommender_metrics_file, 'ranking_metrics.pkl')
        if os.path.exists(ranking_metrics_path):
            with open(ranking_metrics_path, 'rb') as f:
                recommender_metrics = pickle.load(f)
        else:
            raise FileNotFoundError(f"Recommender metrics file not found: {recommender_metrics_file}")

        return df_data, recommender_metrics

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def setup_training_components(script_args: ScriptArguments, config: GRPOConfig,
                              dataset, reward_funcs: List[Callable]):
    """Initialise the model with LoRA adapters and create the GRPO trainer.

    Args:
        script_args: Parsed command-line arguments.
        config: GRPOConfig with training hyperparameters.
        dataset: HuggingFace Dataset for training.
        reward_funcs: List of reward functions for GRPO.

    Returns:
        Tuple of (GRPOTrainer, LoraConfig).
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up training components...")

    try:
        # Select LoRA config based on model family
        if 'gemma' in script_args.model_name: # type: ignore
            lora_config = LoraConfig(**GEMMA_LORA_CONFIG)
        elif 'Llama' in script_args.model_name: # type: ignore
            lora_config = LoraConfig(**LLAMA_LORA_CONFIG)
        elif 'Qwen' in script_args.model_name: # type: ignore
            lora_config = LoraConfig(**QWEN_LORA_CONFIG)
        else:
            raise ValueError(f"Unsupported model: {script_args.model_name}")

        # Load pretrained model in bfloat16 and apply LoRA
        model_config = AutoConfig.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            config=model_config,
            cache_dir=script_args.cache_dir,
            #attn_implementation='eager'  # required for Gemma models
        )
        model = get_peft_model(model, lora_config)

        # Create the GRPO trainer with multiple reward functions
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=config,
            train_dataset=dataset
        )

        return grpo_trainer, lora_config

    except Exception as e:
        logger.error(f"Error setting up training components: {e}")
        raise

# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def reward_func(completions, **kwargs) -> list[float]:
    """Recommendation quality reward based on nDCG@10 relative to baseline."""
    responses = [_coerce_to_text(completion[0]['content']) for completion in completions]
    expected_counts = kwargs["expected_count"]
    ground_truths = kwargs["ground_truth"]
    recommender_scores = kwargs["recommender_score"]

    rewards = []
    for response, expected_count, ground_truth, recommender_score in zip(
        responses, expected_counts, ground_truths, recommender_scores
    ):
        rewards.append(compute_recommendation_quality(
            response, expected_count, ground_truth, recommender_score, metric='ndcg@10'
        ))
    return rewards


def reward_func_rouge(completions, **kwargs) -> list[float]:
    """ROUGE-L reward comparing generated ranking text to ground-truth ordering."""
    responses = [_coerce_to_text(completion[0]['content']) for completion in completions]
    expected_counts = kwargs["expected_count"]
    answers = kwargs["answers"]

    rewards = []
    for response, expected_count, answer in zip(responses, expected_counts, answers):
        rewards.append(compute_rogue_reward(
            response, expected_count, answer, metric='rogueL', rouge=rouge
        ))
    return rewards


def reward_func_reasoning(completions, **kwargs) -> list[float]:
    """Reward that encourages sufficiently detailed candidate news analysis.

    Gives full reward (1.0) for analyses >= 200 characters, with a linear
    scale from -0.5 (empty) to 1.0 (200 chars) for shorter analyses.
    """
    responses = [_coerce_to_text(completion[0]['content']) for completion in completions]
    rewards = []
    for response in responses:
        len_response = lenght_reasoning_content(response, field="candidate_news_analysis")
        if len_response >= 200:
            rewards.append(1.0)
        else:
            # Linear interpolation: 0 chars → -0.50, 200 chars → 1.00
            rewards.append(0.0075 * len_response - 0.5)
    return rewards


def reward_func_json(completions, **kwargs) -> list[float]:
    """Reward for valid JSON output with required fields."""
    responses = [_coerce_to_text(completion[0]['content']) for completion in completions]
    rewards = [
        check_json_output_format(response, required_fields=["user_summary", "candidate_news_analysis", "ranking"])[0]
        for response in responses
    ]
    return rewards

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Entry point: parse args, build dataset, train with GRPO, and save the model."""
    try:
        logger = setup_logging()
        logger.info("Initializing GRPO training...")

        # Parse command-line arguments
        parser = HfArgumentParser(ScriptArguments) # type: ignore
        script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

        # Configure environment
        setup_environment(script_args)

        logger.info(f"Model: {script_args.model_name}")
        logger.info(f"Prompt version: {script_args.prompt_version}")
        logger.info(f"Reward version: {script_args.reward_type}")
        logger.info(f"Recommender metric: {script_args.recommender_metric}")
        logger.info(f"Recommender rewards file: {script_args.recommender_rewards_file}")

        # Validate arguments
        validate_script_args(script_args)

        # Initialise results manager for structured output
        results_manager = TrainingResultsManager(script_args, base_dir=script_args.output_dir)
        logger.info(f"Results will be saved to: {results_manager.run_dir}")

        # Initialise Weights & Biases run
        wandb.require("service")
        wandb.init(
            project=f"{script_args.model_name.split('/')[-1].replace('/', '-')}",
            name=(
                f"{script_args.optimization_algorithm}_prompt-{script_args.prompt_version}"
                f"_reward-{script_args.reward_type}"
                f"_recommender-{script_args.recommender_rewards_file.split('_')[0]}"
                f"_metric-{script_args.recommender_metric}"
            ),
            config=script_args.__dict__
        )

        # Load training data and recommender baseline scores
        train_df, recommender_metrics = load_data(script_args)

        # Create tokenizer (needed before dataset construction)
        logger.info("Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)

        # Build dataset based on chosen metric
        logger.info("Building dataset...")
        if script_args.recommender_metric == "ndcg@10":
            logger.info("Building nDCG@10 dataset...")
            ids, chat_prompts, histories, candidates, history_ids, candidates_ids, labels = \
                build_chat_template_dataset(train_df[:script_args.num_samples], tokenizer, training=True)
            logger.info(f"Number of samples: {len(ids)}")

            expected_counts = [len(c_ids) for c_ids in candidates_ids]
            ground_truths = [histories[i] for i in range(len(histories))]
            recommender_scores = [recommender_metrics[id_] for id_ in ids]

            logger.info(f"Expected counts: {len(expected_counts)}, Recommender scores: {len(recommender_scores)}")
            logger.info(f"Labels: {len(labels)}, Chat prompts: {len(chat_prompts)}")

            tmp_train_df = pd.DataFrame({
                'id': ids,
                'prompt': chat_prompts,
                'expected_count': expected_counts,
                'ground_truth': labels,
                'recommender_score': recommender_scores
            })
            train_dataset = Dataset.from_pandas(tmp_train_df)

        elif script_args.recommender_metric == "rouge":
            logger.info("Building ROUGE dataset...")
            ids, prompts, answers = build_rogue_dataset(
                train_df[:script_args.num_samples], tokenizer, training=True
            )
            logger.info(f"Number of samples: {len(ids)}")

            expected_counts = [len(answer.split(' ')) for answer in answers]
            tmp_train_df = pd.DataFrame({
                'id': ids,
                'prompt': prompts,
                'expected_count': expected_counts,
                'answer': answers
            })
            train_dataset = Dataset.from_pandas(tmp_train_df)

        # Generation parameters for sampling completions
        generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "temperature": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 600
        }

        # Assemble GRPO training configuration
        logger.info("Creating GRPO config...")
        training_args = GRPOConfig(
            # Batching
            generation_batch_size=script_args.generation_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            num_generations=script_args.num_generations,
            torch_empty_cache_steps=script_args.torch_empty_cache_steps,
            num_iterations=script_args.num_iterations,
            # Training
            learning_rate=script_args.learning_rate,
            num_train_epochs=script_args.num_train_epochs,
            seed=script_args.seed,
            remove_unused_columns=script_args.remove_unused_columns,
            reward_weights=[1/3, 1/3, 1/3],  # equal weight: nDCG, JSON format, reasoning
            # Generation
            generation_kwargs=generation_kwargs,
            max_prompt_length=script_args.max_prompt_length,
            disable_tqdm=script_args.disable_tqdm,
            # Logging & checkpointing
            report_to=script_args.report_to,
            log_completions=script_args.log_completions,
            logging_steps=script_args.logging_steps,
            output_dir=results_manager.run_dir,
            save_strategy=script_args.save_strategy,
            save_steps=script_args.save_steps,
        )

        # Build trainer with three reward functions
        logger.info("Creating GRPO trainer...")
        grpo_trainer, lora_config = setup_training_components(
            script_args, training_args, train_dataset,
            [reward_func, reward_func_json, reward_func_reasoning]
        )

        # Persist all configurations for reproducibility
        results_manager.save_config(
            script_args=script_args,
            algorithm_config=training_args,
            lora_config=lora_config,
            generation_kwargs=generation_kwargs
        )

        # Run training
        logger.info("Starting training...")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
        grpo_trainer.train()
        logger.info("Training completed successfully!")

        # Save the final model checkpoint
        os.makedirs(os.path.join(results_manager.run_dir, 'final_model'), exist_ok=True)
        grpo_trainer.save_model(os.path.join(results_manager.run_dir, 'final_model'))
        logger.info(f"Results saved to: {results_manager.run_dir}")

        wandb.finish()

    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
