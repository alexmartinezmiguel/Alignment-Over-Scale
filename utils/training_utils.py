import re
import json
from typing import Optional, List, Text, Tuple, Dict, Any

import torch
from torch import Tensor
import numpy as np
import evaluate # type: ignore
from recommenders.models.deeprec.deeprec_utils import cal_metric # type: ignore


def analyze_ranking_quality(news_ids: List[str], expected_count: int = 10) -> dict:
    """
    Analyze the quality of extracted ranking.
    
    Args:
        news_ids (List[str]): List of extracted news IDs
        
    Returns:
        dict: Quality metrics
    """
    quality_metrics = {
        'count': len(news_ids),
        'expected_count': expected_count,
        'has_placeholders': 'C#' in news_ids,
        'has_duplicates': len(news_ids) != len(set(news_ids)),
        'is_sequential': False,
        'max_id': 0,
        'confidence': 'low'
    }
    
    # Check if ranking has valid IDs
    valid_ids = [nid for nid in news_ids if nid != 'C#']
    if valid_ids:
        # Extract numbers and check sequence
        try:
            numbers = [int(nid[1:]) for nid in valid_ids]
            quality_metrics['max_id'] = max(numbers)
            
            # Check if it's a reasonable sequential range
            if len(set(numbers)) == len(numbers) == expected_count:
                quality_metrics['is_sequential'] = True
                quality_metrics['confidence'] = 'high'
        except ValueError:
            pass
    
    # Adjust confidence based on issues
    if quality_metrics['has_placeholders'] or quality_metrics['has_duplicates']:
        quality_metrics['confidence'] = 'low'
    
    return quality_metrics

def is_sequential_ranking(ranking: List[str]) -> bool:
    """
    Check if the ranking is in sequential order (C1, C2, C3, ...).
    Rankings with length 2, 3 or 4 are not considered sequential for penalty purposes.
    
    Args:
        ranking (List[str]): List of news IDs
        
    Returns:
        bool: True if ranking is in sequential order starting from C1 and length > 3
    """
    if not ranking:
        return False
    
    # Don't penalize short rankings (length 2, 3 or 4)
    if len(ranking) <= 4:
        return False
    
    try:
        # Extract numbers from the ranking
        numbers = []
        for news_id in ranking:
            if news_id.startswith('C') and news_id[1:].isdigit():
                numbers.append(int(news_id[1:]))
            else:
                return False  # Invalid format
        
        # Check if it's sequential starting from 1
        expected_sequence = list(range(1, len(numbers) + 1))
        return numbers == expected_sequence
        
    except (ValueError, IndexError):
        return False



def contains_invalid_h_ids(ranking: List[str]) -> bool:
    """
    Check if the ranking contains invalid H# IDs (like H1, H2, etc.).
    
    Args:
        ranking (List[str]): List of news IDs
        
    Returns:
        bool: True if ranking contains H# patterns
    """
    if not ranking:
        return False
    
    # Pattern to match H followed by digits
    h_pattern = r'^H\d+$'
    
    for news_id in ranking:
        if re.match(h_pattern, news_id):
            return True
    
    return False



def check_json_output_format(text: str, required_fields: List[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Simple format checker for reasoning outputs.
    
    Checks for:
    1. Valid JSON.
    2. Presence of all fields specified in the `required_fields` list.
    3. Specific type/format for "user summary", "reasoning", and "ranking" 
       fields *ONLY if they are listed in required_fields*.
    
    Returns a reward and detailed feedback.
    - Returns -1.0 for JSON errors or missing required fields.
    - Starts with 1.0 and subtracts penalties (e.g., 0.3) for type errors.
    """
    feedback = {}
    reward = 1.0

    # Try to parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        feedback["json_error"] = str(e)
        return -1.0, feedback

    # Check for required fields
    missing = [f for f in required_fields if f not in data]
    if missing:
        feedback["missing_fields"] = f"Missing fields: {', '.join(missing)}"
        return -1.0, feedback

    # Check "user summary" *if it was required*
    if "user_summary" in required_fields:
        if not isinstance(data["user_summary"], str) or not data["user_summary"].strip():
            feedback["user_summary_error"] = "Field 'user_summary' must be a non-empty string."
            reward -= 0.3

    # Check "candidate_news_analysis" *if it was required*
    if "candidate_news_analysis" in required_fields:
        if not isinstance(data["candidate_news_analysis"], (str, list, dict)):
            feedback["candidate_news_analysis_error"] = "Field 'candidate_news_analysis' must be a non-empty string or a list or a dict."
            reward -= 0.3

    # Check "ranking" *if it was required*
    if "ranking" in required_fields:
        ranking = data["ranking"]
        if not isinstance(ranking, list):
            feedback["ranking_error"] = "'ranking' must be a list."
            reward -= 0.3 #0.9 if required_fields = ["ranking"]

    # ✅ If no errors were found
    if not feedback:
        feedback["status"] = "Valid format"
        
    return reward, feedback


def compute_recommendation_quality(
    text: str,
    expected_count: int = 10,
    ground_truth: str = None,
    recommender_scores: dict = None,
    metric: str = "ndcg@10"
) -> float:
    """
    Reward function for RL fine-tuning of LLM-based news recommendation.

    Args:
        text (str): LLM output containing the ranking and reasoning.
        expected_count (int): Number of candidate items expected in ranking.
        ground_truth (str): Comma-separated IDs of clicked/positive items.
        recommender_scores (dict): Baseline recommender scores for comparison.
        metric (str): Ranking metric to evaluate (default: 'ndcg@10').

    Returns:
        float: Reward value in [0, 2.1].
    """

    # --- Extract and validate format ---
    format_feedback = check_json_output_format(text, required_fields = ["ranking"])
    if format_feedback[1].get('json_error') or format_feedback[1].get('missing_fields') or format_feedback[1].get('ranking_error'):
        return 0.0
    
    # for correct formatted outputs
    data = text if isinstance(text, (dict,list)) else json.loads(text)
    ranking = [str(x) for x in data['ranking']]
    if contains_invalid_h_ids(ranking):
        return -1.0  # final ranking contains items from the history

    quality = analyze_ranking_quality(ranking, expected_count)
    if (
        quality["count"] != quality["expected_count"]
        or quality["has_duplicates"]
        or quality["has_placeholders"]
    ):
        return -0.5  # format almost correct but flawed

    # --- Penalize trivial sequential order (C1, C2, C3, ...) ---
    if is_sequential_ranking(ranking):
        return -0.8  # discourage trivial outputs without collapsing PPO

    try:
        # --- Build labels and predictions for metric evaluation ---
        preds = np.array([len(ranking) - i for i in range(len(ranking))])
        indices = [ranking.index(l) for l in ground_truth.split(",")]
        labels = np.array([1 if i in indices else 0 for i in range(len(ranking))])

        llm_scores = cal_metric([labels], [preds], ["ndcg@5;10"])
        ndcg_llm = llm_scores[metric]
        ndcg_base = recommender_scores[metric]

        # --- Relative scaling vs baseline ---
        reward = ndcg_llm / (ndcg_base + 1e-8)  # baseline = 1.0 reference
        reward = max(0.0, min(2.0, reward))     # clamp to [0,2]
        return reward

    except Exception:
        return -0.5  # recoverable error (e.g. label not found in ranking)

# v5 json format + recommendation quality + length of the reasoning content (weights = 0.3, 0.5, 0.2)
def lenght_reasoning_content(text: str, field: str = "candidate_news_analysis") -> int:
    """
    Calculate the length of the reasoning content in the generated text.
    """
    # --- Extract and validate format ---
    format_feedback = check_json_output_format(text, required_fields = [field])
    if format_feedback[1].get('json_error') or format_feedback[1].get('missing_fields') or format_feedback[1].get(f'{field}_error'):
        return 0.0
    
    data_dict = text if isinstance(text, (dict,list)) else json.loads(text)
    data_field = data_dict[field]
    if not isinstance(data_field, (str,dict)):
        return 0
    return len(str(data_field))

def _coerce_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # common cases: {'type':'text','text': '...'} or OpenAI-like message parts
        if 'text' in content and isinstance(content['text'], str):
            return content['text']
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = []
        for el in content:
            if isinstance(el, dict) and 'text' in el and isinstance(el['text'], str):
                parts.append(el['text'])
            else:
                parts.append(str(el))
        return "\n".join(parts)
    return str(content)




def compute_rogue_reward(
    text: str,
    expected_count: int = 10,
    ground_truth: str = None,
    metric: str = "rougeL",
    rouge: evaluate.load = None
) -> float:
    """
    Reward function for RL fine-tuning of LLM-based news recommendation.

    Args:
        text (str): LLM output containing the ranking and reasoning.
        expected_count (int): Number of candidate items expected in ranking.
        ground_truth (str): Comma-separated IDs of clicked/positive items.
        metric (str): Rogue metric to evaluate (default: 'rogueL').

    Returns:
        float: Reward value in [0, 2.1].
    """

    # --- Extract and validate format ---
    format_feedback = check_json_output_format(text, required_fields = ["ranking"])
    if format_feedback[1].get('json_error') or format_feedback[1].get('missing_fields') or format_feedback[1].get('ranking_error'):
        return 0.0
    
    # for correct formatted outputs
    data = text if isinstance(text, (dict,list)) else json.loads(text)
    ranking = [str(x) for x in data['ranking']]
    if contains_invalid_h_ids(ranking):
        return -1.0  # final ranking contains items from the history

    quality = analyze_ranking_quality(ranking, expected_count)
    if (
        quality["count"] != quality["expected_count"]
        or quality["has_duplicates"]
        or quality["has_placeholders"]
    ):
        return -0.5  # format almost correct but flawed

    # --- Penalize trivial sequential order (C1, C2, C3, ...) ---
    if is_sequential_ranking(ranking):
        return -0.8  # discourage trivial outputs without collapsing PPO

    try:
        generated_text = (' ').join(ranking)
        reward = rouge.compute(
            predictions=[generated_text], 
            references=[ground_truth], 
            rouge_types=[metric], 
            use_aggregator=False, 
            use_stemmer=False
        )

        return reward[metric][0]

    except Exception:
        return -0.5  # recoverable error (e.g. label not found in ranking)