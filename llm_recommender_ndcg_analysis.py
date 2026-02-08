#!/usr/bin/env python3
"""
Evaluate fine-tuned LLM checkpoints on the MIND test set using ranking metrics
(nDCG@5, nDCG@10, AUC, MRR).

This script iterates over model checkpoints saved during training, generates
ranked recommendation lists for each test impression, and computes standard
information-retrieval metrics. Results are saved incrementally to CSV files so
the evaluation can be resumed.

Usage:
    python llm_recommender_ndcg_analysis.py \
        --model-type google/gemma-2b-it \
        --batch-size 4
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import gc
import json
import random
import argparse

import numpy as np # type: ignore
import torch # type: ignore
import pandas as pd # type: ignore
import datasets # type: ignore
import tensorflow as tf # type: ignore
from datasets import Dataset # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

# Set working directory for Docker container execution
os.chdir('/workspace')

# Custom utilities
from utils.data_utils import build_chat_template_dataset
from utils.misc_utils import setup_logging, str2bool
from utils.training_utils import _coerce_to_text, analyze_ranking_quality, contains_invalid_h_ids
from recommenders.models.deeprec.deeprec_utils import cal_metric # type: ignore

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

logger = setup_logging()
datasets.disable_progress_bar()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'  # Replace with your own Hugging Face token
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # required for Gemma models
os.environ['HF_HOME'] = 'hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'hf_cache/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'hf_cache/hub'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

cache_dir = 'hf_cache/hub'
os.makedirs(cache_dir, exist_ok=True)

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM checkpoints on news recommendation ranking metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-type', type=str, default='mind',
                        help='Dataset to use (e.g. "mind")')
    parser.add_argument('--model-type', type=str, default='google/gemma-2b-it',
                        help='Model identifier (used to locate checkpoint directories)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for inference')
    return parser.parse_args()

# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def main():
    """Iterate over checkpoints, generate rankings, and compute metrics."""
    args = parse_arguments()

    # Verify GPU availability
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices or 'GPU' not in str(gpu_devices[0]):
        logger.error('GPU not detected')
    logger.info(f"Using model {args.model_type} for nDCG analysis...")

    # Paths for input data and output results
    data_path = os.path.join(os.getcwd(), f"data/{args.data_type}/processed_recommender/")
    test_file = os.path.join(data_path, 'MIND_test.csv')
    test_df = pd.read_csv(test_file, sep=',', header=0)

    output_dir = os.path.join(
        data_path,
        f"personalized_dataset_{args.model_type.split('/')[-1]}-temp-1.0-ndcg-analysis"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Accumulators for metrics across checkpoints
    ndcg_5, ndcg_10, auc, mrr = [], [], [], []
    checkpoints_completed = []

    # ----- Loop over training checkpoints -----
    for checkpoint in range(30000, 49100, 100):
        model_path = os.path.join(
            os.getcwd(),
            f'output/llm_finetuning/runs/{args.model_type}/checkpoint-{checkpoint}'
        )
        logger.info(f"Loading tokenizer and model from checkpoint {checkpoint}...")

        # Load tokenizer and model for this checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', cache_dir=cache_dir)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=cache_dir,
            attn_implementation='eager'  # required for Gemma models
        )

        # Ensure pad token is set for batched generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build test dataset (chat-template formatted prompts)
        ids, dataset = build_chat_template_dataset(test_df, tokenizer)

        # Generation hyperparameters
        generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "temperature": 1.0,
            "do_sample": True,
            "max_new_tokens": 800
        }

        # ----- Generate recommendations in batches -----
        recommendations_list = []
        for i in range(0, len(dataset), args.batch_size):
            batch_end = min(i + args.batch_size, len(dataset))
            batch_ids = ids[i:batch_end]
            batch_messages = dataset[i:batch_end]

            # Apply chat template and tokenize
            batch_formatted_messages = tokenizer.apply_chat_template(
                batch_messages, padding=True, tokenize=False, add_generation_prompt=True
            )
            tokenized_batch = tokenizer(
                batch_formatted_messages, truncation=False, return_tensors="pt", padding=True
            )

            # Generate responses
            with torch.no_grad():
                batch_outputs = llm_model.generate(
                    tokenized_batch['input_ids'].to('cuda'),
                    attention_mask=tokenized_batch['attention_mask'].to('cuda'),
                    **generation_kwargs
                )

            # Free GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Decode generated tokens (strip input prompt)
            input_lengths = tokenized_batch['input_ids'].shape[1]
            for j, output in enumerate(batch_outputs):
                generated_tokens = output[input_lengths:]
                decoded_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                recommendations_list.append(decoded_text)
                logger.info(f"Impression ID: {batch_ids[j]}, Generation: {decoded_text}")

        # ----- Parse generated rankings -----
        generated_text = pd.DataFrame({
            'impression_id': ids,
            'recommendations': recommendations_list
        })

        ranked_lists = {}
        ranked_lists_quality = {}

        for index, row in generated_text.iterrows():
            recommendations = row.iloc[1]
            recommendation_list = None

            # Attempt JSON parsing with progressive fallbacks
            try:
                recommendation_list = json.loads(_coerce_to_text(recommendations))
            except json.JSONDecodeError:
                text = _coerce_to_text(recommendations).rstrip()

                # Try fixing unbalanced braces/brackets
                open_brackets = text.count('[') - text.count(']')
                open_braces = text.count('{') - text.count('}')
                fixed_text = text + ']' * open_brackets + '}' * open_braces

                try:
                    recommendation_list = json.loads(fixed_text)
                except json.JSONDecodeError:
                    # Regex fallback: extract ranking array directly
                    match = re.search(r'"ranking"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                    if match:
                        try:
                            ranking_str = '[' + match.group(1) + ']'
                            ranking = json.loads(ranking_str)
                            recommendation_list = {'ranking': ranking}
                        except json.JSONDecodeError:
                            pass

            # Fallback to random ranking if parsing failed
            if recommendation_list is None or 'ranking' not in recommendation_list:
                recommendation_list = {'ranking': [f"C{k}" for k in range(1, 11)]}
                random.shuffle(recommendation_list['ranking'])

            quality = analyze_ranking_quality(recommendation_list['ranking'])
            ranked_lists.setdefault(row.iloc[0], []).append(recommendation_list['ranking'])
            ranked_lists_quality.setdefault(row.iloc[0], {}).update(quality)

        # ----- Compute ranking metrics -----
        groups = []
        labels = []

        for impression_id, recommendations in ranked_lists.items():
            if contains_invalid_h_ids(recommendations[0]):
                # Invalid ranking: assign random scores
                group_preds = [float(10 - k) for k in range(10)]
                group_labels = np.zeros(10, dtype=np.int32)
                group_labels[random.randint(0, 9)] = 1
                groups.append(np.array(group_preds, dtype=np.float32))
                labels.append(group_labels)
            else:
                quality = ranked_lists_quality[impression_id]
                if quality['count'] == 10 and not quality['has_duplicates'] and recommendations[0]:
                    label = test_df.loc[test_df['impression_id'] == impression_id, 'label'].values[0]
                    group_preds = [float(len(recommendations[0]) - k) for k in range(len(recommendations[0]))]
                    try:
                        label_index = recommendations[0].index(label)
                        group_labels = [0 if k != label_index else 1 for k in range(len(recommendations[0]))]
                        groups.append(np.array(group_preds, dtype=np.float32))
                        labels.append(np.array(group_labels, dtype=np.int32))
                    except ValueError:
                        # Label not found in ranking: assign random scores
                        group_preds = [float(len(recommendations[0]) - k) for k in range(10)]
                        group_labels = np.zeros(10)
                        group_labels[random.randint(0, 9)] = 1
                        groups.append(np.array(group_preds, dtype=np.float32))
                        labels.append(group_labels)
                else:
                    # Low-quality ranking: assign random scores
                    group_preds = [float(10 - k) for k in range(10)]
                    group_labels = np.zeros(10, dtype=np.int32)
                    group_labels[random.randint(0, 9)] = 1
                    groups.append(np.array(group_preds, dtype=np.float32))
                    labels.append(group_labels)

        # Compute nDCG@5, nDCG@10, AUC, and MRR
        res = cal_metric(labels, groups, ['group_auc', 'mean_mrr', 'ndcg@5;10'])
        ndcg_5.append(res['ndcg@5'])
        ndcg_10.append(res['ndcg@10'])
        auc.append(res['group_auc'])
        mrr.append(res['mean_mrr'])

        # Save results incrementally (one row per checkpoint)
        checkpoints_completed.append(checkpoint)
        pd.DataFrame({'checkpoint': checkpoints_completed, 'ndcg@5': ndcg_5}).to_csv(
            os.path.join(output_dir, 'ndcg_5.csv'), index=False)
        pd.DataFrame({'checkpoint': checkpoints_completed, 'ndcg@10': ndcg_10}).to_csv(
            os.path.join(output_dir, 'ndcg_10.csv'), index=False)
        pd.DataFrame({'checkpoint': checkpoints_completed, 'auc': auc}).to_csv(
            os.path.join(output_dir, 'auc.csv'), index=False)
        pd.DataFrame({'checkpoint': checkpoints_completed, 'mrr': mrr}).to_csv(
            os.path.join(output_dir, 'mrr.csv'), index=False)


if __name__ == "__main__":
    main()
