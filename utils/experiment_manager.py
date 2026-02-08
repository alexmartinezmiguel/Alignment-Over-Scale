import os
import json
import pickle
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import torch

class TrainingResultsManager:
    """
    Manages the directory structure and storage of training results.
    
    Directory Structure:
    pens_output/LLM_checkpoints/
    ├── runs/
    │   ├── gemma-2b-it_prompt-v3_reward-v3_alpha-1.0_batch-v1/
    │   │   ├── config/
    │   │   │   ├── script_args.json
    │   │   │   ├── ppo_config.json
    │   │   │   ├── lora_config.json
    │   │   │   └── training_metadata.json
    │   │   ├── models/
    │   │   │   └── final_model/
    │   │   ├── metrics/
    │   │   │   ├── training_stats.pkl (updated after each batch)
    │   │   │   ├── batch_metrics.csv (updated after each batch)
    │   │   │   └── plots/
    │   │   ├── generated_content/
    │   │   │   ├── rewritten_articles.pkl (updated after each batch)
    │   │   │   └── sample_generations.txt (updated after each batch)
    │   │   └── README.md
    │   └── ...
    ├── logs/
    """
    
    def __init__(self, script_args, base_dir: str = "pens_output/LLM_checkpoints", 
                 log_file: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.script_args = script_args
        self.log_file = log_file  # Store log file path for metadata
        
        # Create descriptive run name from experiment parameters
        self.run_name = self._create_run_name(script_args)
        
        # Create directory structure
        self.run_dir = self.base_dir / "runs" / self.run_name
        self.setup_directories()
        
        # Initialize metrics storage
        self.metrics_file = self.run_dir / "metrics" / "batch_metrics.csv"
        self.stats_file = self.run_dir / "metrics" / "training_stats.pkl"
        self.articles_file = self.run_dir / "generated_content" / "generations.pkl"
        
    def _create_run_name(self, script_args) -> str:
        """Create a descriptive run name from experiment parameters."""
        # Extract model name (remove path and special characters)
        model_name = script_args.model_name.split('/')[-1].replace('/', '-')
        
        # Create descriptive name
        run_name = (f"{model_name}_"
                   f"prompt-{script_args.prompt_version}_"
                   f"reward-{script_args.reward_type}_"
                   f"recommender-metric_{script_args.recommender_metric}_"
                   f"full-training_num-generations-{script_args.num_generations}")

        # Add additional distinguishing features if needed
        if script_args.learning_rate != 1.41e-5:  # Non-default learning rate
            run_name += f"_lr-{script_args.learning_rate}"
        
        return run_name
    
    def setup_directories(self):
        """Create the directory structure for the current run."""
        directories = [
            self.run_dir / "config",
            self.run_dir / "final_model",
            self.run_dir / "metrics" / "plots",
            self.run_dir / "generated_content"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _make_json_serializable(self, obj):
        """Convert non-JSON serializable objects to serializable forms."""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, torch.device):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj

    def save_config(self, script_args, algorithm_config, lora_config=None, 
                early_stopping_config=None, generation_kwargs=None):
        """Save all configuration parameters."""
        config_dir = self.run_dir / "config"
        
        # Save script arguments
        script_args_dict = {
            k: self._make_json_serializable(v) 
            for k, v in script_args.__dict__.items()
        }
        with open(config_dir / "script_args.json", "w") as f:
            json.dump(script_args_dict, f, indent=2)
        
        # Save PPO config
        algorithm_config_dict = {
            k: self._make_json_serializable(v) for k, v in algorithm_config.__dict__.items() 
            if not k.startswith('_') and v is not None
        }
        with open(config_dir / "algorithm_config.json", "w") as f:
            json.dump(algorithm_config_dict, f, indent=2)
        
        # Save LoRA config if provided
        if lora_config:
            lora_config_dict = {
                k: self._make_json_serializable(v) for k, v in lora_config.__dict__.items() 
                if not k.startswith('_')
            }
            with open(config_dir / "lora_config.json", "w") as f:
                json.dump(lora_config_dict, f, indent=2)
        
        # Save additional configurations
        metadata = {
            "experiment_name": self.run_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "log_file": str(self.log_file) if self.log_file else None,
            "early_stopping_config": early_stopping_config or {},
            "generation_kwargs": generation_kwargs or {},
            "git_commit": self.get_git_commit(),
            "python_packages": self.get_package_versions()
        }
        
        with open(config_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        import pkg_resources
        key_packages = ['torch', 'transformers', 'trl', 'peft', 'datasets', 
                       'sentence-transformers', 'numpy', 'pandas']
        versions = {}
        for package in key_packages:
            try:
                versions[package] = pkg_resources.get_distribution(package).version
            except:
                versions[package] = "unknown"
        return versions
    
    def save_final_model(self, model, tokenizer):
        """Save the final trained model."""
        final_dir = self.run_dir / "models" 
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    
    def update_batch_results(self, batch_idx: int, kl_values: List[float], 
                           mean_rewards: List[float], std_rewards: List[float], 
                           entropy: List[float], mean_non_score_rewards: List[float],
                           generated_content: Dict[str, List[str]]):
        """Update all results after each batch (overwrites previous data)."""
        
        # Save detailed metrics (overwrite each time)
        training_stats = {
            "kl_values": kl_values,
            "mean_rewards": mean_rewards,
            "std_rewards": std_rewards,
            "entropy": entropy,
            "mean_non_score_rewards": mean_non_score_rewards,
            "total_batches": len(mean_rewards),
            "current_batch": batch_idx + 1,
            "final_metrics": {
                "final_kl": kl_values[-1] if kl_values else 0,
                "final_mean_reward": mean_rewards[-1] if mean_rewards else 0,
                "final_std_reward": std_rewards[-1] if std_rewards else 0,
                "final_entropy": entropy[-1] if entropy else 0
            }
        }
        
        with open(self.stats_file, "wb") as f:
            pickle.dump(training_stats, f)
        
        # Save as CSV for easy analysis (overwrite each time)
        metrics_df = pd.DataFrame({
            "batch": range(len(mean_rewards)),
            "kl_divergence": kl_values,
            "mean_reward": mean_rewards,
            "std_reward": std_rewards,
            "entropy": entropy,
            "mean_non_score_reward": mean_non_score_rewards
        })
        metrics_df.to_csv(self.metrics_file, index=False)
        
        # Save generated content (overwrite each time)
        with open(self.articles_file, "wb") as f:
            pickle.dump(generated_content, f)