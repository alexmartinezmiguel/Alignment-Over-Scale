#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script for LLM-based news recommendation on the MIND dataset.

This script fine-tunes a pre-trained language model using SFT to generate
ranked news recommendations. It supports multiple model architectures
(Gemma, Llama, Qwen) with LoRA adapters for parameter-efficient training.

Usage:
    python finetune_llm_sft.py \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --prompt_version v8 \
        --num_train_epochs 20
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import logging
from typing import Optional
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
from trl import SFTConfig, SFTTrainer # type: ignore
from peft import LoraConfig, get_peft_model # type: ignore
import wandb # type: ignore

# Set working directory for Docker container execution
os.chdir('/workspace')

# Custom utilities
from utils.misc_utils import setup_logging # type: ignore
from utils.data_utils import build_sft_dataset # type: ignore
from utils.experiment_manager import TrainingResultsManager # type: ignore

# =============================================================================
# CONSTANTS
# =============================================================================

# API tokens — replace with your own credentials
HF_TOKEN = 'YOUR_HF_TOKEN_HERE'  # Replace with your own Hugging Face token
WANDB_API_KEY = 'YOUR_WANDB_API_KEY_HERE'  # Replace with your own Weights & Biases API key

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
    """Command-line arguments for SFT training."""

    # Model selection
    optimization_algorithm: Optional[str] = field(default="SFT", metadata={"help": "the optimization algorithm"})
    model_name: Optional[str] = field(default="meta-llama/Llama-3.2-3B-Instruct", metadata={"help": "the model name"})

    # Training hyperparameters
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    num_train_epochs: Optional[int] = field(default=20, metadata={"help": "the number of training epochs"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": "whether to remove unused columns"})

    # Batching parameters
    gradient_accumulation_steps: Optional[int] = field(default=12, metadata={"help": "the number of gradient accumulation steps"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "the per device train batch size"})
    torch_empty_cache_steps: Optional[int] = field(default=1, metadata={"help": "the number of steps to empty the cache"})

    # Generation parameters
    generation_kwargs: Optional[dict] = field(default=None, metadata={"help": "the generation kwargs"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "maximum length of the tokenized sequence"})
    disable_tqdm: Optional[bool] = field(default=True, metadata={"help": "whether to disable the tqdm"})

    # Logging and checkpointing
    report_to: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the logging steps"})
    output_dir: Optional[str] = field(default=os.path.join(os.getcwd(), 'output/llm_finetuning'), metadata={"help": "path to the output directory"})
    save_strategy: Optional[str] = field(default='steps', metadata={"help": "the save strategy"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the save steps"})

    # Data parameters
    data_path: Optional[str] = field(default=os.path.join(os.getcwd(), 'data/mind/processed_recommender'), metadata={"help": "path to the data"})
    cache_dir: Optional[str] = field(default='hf_cache/hub', metadata={"help": "path to cache LLM models"})
    train_dataset: Optional[str] = field(default="MIND_train.csv", metadata={"help": "the dataset name"})
    num_samples: Optional[int] = field(default=15000, metadata={"help": "the number of samples to use for training"})
    prompt_version: Optional[str] = field(default="v8", metadata={"help": "the version of the prompt"})

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


def load_data(script_args: ScriptArguments) -> pd.DataFrame:
    """Load the training CSV file.

    Args:
        script_args: Parsed command-line arguments containing data path info.

    Returns:
        DataFrame with training samples.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")

    try:
        data_file = Path(script_args.data_path) / script_args.train_dataset # type: ignore
        df_data = pd.read_csv(data_file)
        return df_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def setup_training_components(script_args: ScriptArguments, config: SFTConfig, dataset):
    """Initialise the model with LoRA adapters and create the SFT trainer.

    Selects the appropriate LoRA configuration based on the model family,
    loads the pretrained model, applies LoRA, and wraps everything in an
    SFTTrainer.

    Args:
        script_args: Parsed command-line arguments.
        config: SFTConfig with training hyperparameters.
        dataset: HuggingFace Dataset for training.

    Returns:
        Tuple of (SFTTrainer, LoraConfig).
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
            cache_dir=script_args.cache_dir
        )
        model = get_peft_model(model, lora_config)

        # Create the SFT trainer
        sft_trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset
        )

        return sft_trainer, lora_config

    except Exception as e:
        logger.error(f"Error setting up training components: {e}")
        raise

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Entry point: parse args, build dataset, train, and save the model."""
    try:
        logger = setup_logging()
        logger.info("Initializing SFT training...")

        # Parse command-line arguments
        parser = HfArgumentParser(ScriptArguments) # type: ignore
        script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

        # Configure environment
        setup_environment(script_args)

        logger.info(f"Model: {script_args.model_name}")
        logger.info(f"Prompt version: {script_args.prompt_version}")

        # Validate arguments
        validate_script_args(script_args)

        # Initialise results manager for structured output
        results_manager = TrainingResultsManager(script_args, base_dir=script_args.output_dir)
        logger.info(f"Results will be saved to: {results_manager.run_dir}")

        # Initialise Weights & Biases run
        wandb.init(
            project=f"{script_args.model_name.split('/')[-1].replace('/', '-')}",
            name=f"{script_args.optimization_algorithm}_prompt-{script_args.prompt_version}",
            config=script_args.__dict__
        )

        # Load training data
        train_df = load_data(script_args)

        # Create tokenizer (needed before dataset construction)
        logger.info("Setting up tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)

        # Build SFT dataset (prompt + completion pairs)
        logger.info("Building dataset...")
        ids, prompts, completions = build_sft_dataset(train_df[:script_args.num_samples])
        logger.info(f"Dataset size — samples: {len(ids)}, prompts: {len(prompts)}, completions: {len(completions)}")

        tmp_train_df = pd.DataFrame({'prompt': prompts, 'completion': completions})
        train_dataset = Dataset.from_pandas(tmp_train_df)

        # Assemble SFT training configuration
        logger.info("Creating SFT config...")
        training_args = SFTConfig(
            # Batching
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            torch_empty_cache_steps=script_args.torch_empty_cache_steps,
            # Training
            learning_rate=script_args.learning_rate,
            num_train_epochs=script_args.num_train_epochs,
            seed=script_args.seed,
            remove_unused_columns=script_args.remove_unused_columns,
            # Generation
            max_length=script_args.max_length,
            disable_tqdm=script_args.disable_tqdm,
            # Logging & checkpointing
            report_to=script_args.report_to,
            logging_steps=script_args.logging_steps,
            output_dir=results_manager.run_dir,
            save_strategy=script_args.save_strategy,
            save_steps=script_args.save_steps,
        )

        # Build trainer with LoRA-adapted model
        logger.info("Creating SFT Trainer...")
        sft_trainer, lora_config = setup_training_components(script_args, training_args, train_dataset)

        # Persist all configurations for reproducibility
        results_manager.save_config(
            script_args=script_args,
            algorithm_config=training_args,
            lora_config=lora_config
        )

        # Run training
        logger.info("Starting training...")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
        sft_trainer.train()
        logger.info("Training completed successfully!")

        # Save the final model checkpoint
        os.makedirs(os.path.join(results_manager.run_dir, 'final_model'), exist_ok=True)
        sft_trainer.save_model(os.path.join(results_manager.run_dir, 'final_model'))
        logger.info(f"Results saved to: {results_manager.run_dir}")

        wandb.finish()

    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
