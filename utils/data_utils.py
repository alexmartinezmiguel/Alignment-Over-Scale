import re
import random
import json
import pandas as pd # type: ignore
from transformers import PreTrainedTokenizerBase # type: ignore
from trl import PPOConfig # type: ignore
from datasets import Dataset # type: ignore


# LLM RECOMMENDER
def build_chat_template_dataset(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, training: bool = False):
    # re expression for data
    history_re = r'H\d+:\s*(.*)'
    candidate_re = r'C\d+:\s*(.*)'
    label_re = r'C(\d+)'

    all_messages = []
    ids = []

    if training:
        all_histories = []
        all_candidates = []
        all_histories_ids = []
        all_candidates_ids = []
        all_labels = []

    # US + R PROMPT
    system_prompt = (
        "You serve as a personalized news recommendation system. Understand the User's History News, and then generate a recommendation list from the Candidate News ranked by how well they match the user's interests (i.e., similarity/continuity with the history). "
        "Instruction: Generate the answer thinking step by step.\n"
        "Output Format: Output your answer strictly in valid JSON with this exact structure and field order:\n\n"
        "{\n"
        '  "user summary": "<summary of the user\'s interests>",\n'
        '  "reasoning": "<reasoning process, analyzing the relevance of the candidate news based on the user summary>",\n'
        '  "ranking": ["C#", "C#", ..., "C#"]\n'
        "}\n\n"
        "Important:\n"
        "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
        "- Every candidate must appear in both the reasoning and the ranking.\n"
        "- The ranking order must match the reasoning.\n"
        "- Do not include anything outside the JSON block."
        )

    # US + CA + R PROMPT
    system_prompt = (
        "You serve as a personalized news recommendation system. Understand the User's History News, and then generate a recommendation list from the Candidate News ranked by how well they match the user's interests (i.e., similarity/continuity with the history)."
        "Output Format: Output your answer strictly in valid JSON with this exact structure and field order:\n\n"
        "{\n"
        '  "user_summary": "<summary of the user\'s interests based on the history>",\n'
        '  "candidate_news_analysis": "<analyze the relevance of each candidate news based on the user interests>",\n'
        '  "ranking": ["C#", "C#", ..., "C#"]\n'
        "}\n\n"
        "Important:\n"
        "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
        "- Every candidate must appear in both the candidate news analysis and the ranking.\n"
        "- The ranking order must match the candidate news analysis.\n"
        "- Do not include anything outside the JSON block."
        )

    # Direct Ranking Prompt
    system_prompt = (
        "You serve as a personalized news recommendation system. Understand the User's History News, and then generate a recommendation list from the Candidate News ranked by how well they match the user's interests (i.e., similarity/continuity with the history)."
        "Output Format: Output your answer strictly in valid JSON with this exact structure and field order:\n\n"
        "{\n"
        '  "ranking": ["C#", "C#", ..., "C#"]\n'
        "}\n\n"
        "Important:\n"
        "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
        "- The ranking must be ordered from most relevant to least relevant.\n"
        "- Do not include anything outside the JSON block."
        )

    for row in df.iterrows():
        # ids.append(row[1]['impression_id'])
        len_candidates = len(row[1]['candidate_news_id'].split('\n'))
        user_prompt = (
            f"User's History News: {row[1]['history']}\n"
            f"Candidate News: {row[1]['candidate']}\n"
            f"Important: Imperative to include all {len_candidates} ids of the Candidate News set.\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]    
        # for gemma models
        # messages = [
        #     {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
        # ]  


        if training:
            # print(len(row[1]['history_news_id'].split('\n')))
            if len(row[1]['history_news_id'].split('\n')) <= 25:
                ids.append(row[1]['impression_id'])
                all_histories.append(re.findall(history_re, row[1]['history']))
                all_candidates.append(re.findall(candidate_re, row[1]['candidate']))
                all_histories_ids.append(row[1]['history_news_id'].split('\n'))
                all_candidates_ids.append(row[1]['candidate_news_id'].split('\n'))
                all_labels.append(row[1]['label'])
                all_messages.append(messages)
        else:
            ids.append(row[1]['impression_id'])
            all_messages.append(messages)

    if training:
        return ids, all_messages, all_histories, all_candidates, all_histories_ids, all_candidates_ids, all_labels
    else:
        return ids, all_messages

def build_sft_dataset(df_data):
    # build SFT dataset
    all_prompts = []
    all_answers = []
    ids = []
    for row in df_data.iterrows():
        ids.append(row[1]['impression_id'])
        len_candidates = len(row[1]['candidate_news_id'].split('\n'))
        # v8 prompt
        system_prompt = (
            f"You serve as a personalized news recommendation system. Understand the User's History News, and then generate a recommendation list from the Candidate News ranked by how well they match the user's interests (i.e., similarity/continuity with the history).\n"
            f"User's History News: {row[1]['history']}\n"
            f"Candidate News: {row[1]['candidate']}\n"
            f"Output Format: Output your answer strictly in valid JSON with this exact structure and field order:\n\n"
            '{"ranking": ["C#", "C#", ..., "C#"]}'
            "Important:\n"
            "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
            f"-Imperative to include all {len_candidates} ids of the Candidate News set.\n"
            "- The ranking must be ordered from most relevant to least relevant.\n"
            "- Do not include anything outside the JSON block."
            )
        # build ground truth sequence
        ground_truths = row[1]['label'].split(",")
        all_candidates = ['C' + str(i) for i in range(1, len_candidates + 1)]
        # remove ground truth candidates from all candidates
        for gt in ground_truths:
            all_candidates.remove(gt)
        # shuffle rest of candidates and append to ground truths
        random.shuffle(all_candidates)
        for c in all_candidates:
            ground_truths.append(c)
        ground_truths_str = ', '.join(ground_truths)
        answer = (
            f'{{"ranking": ["{ground_truths_str}"]}}'
        )
        all_prompts.append(system_prompt)
        all_answers.append(answer)
    
    return ids, all_prompts, all_answers


def build_rogue_dataset(df_data, training: bool = True):

    # re expression for data
    history_re = r'H\d+:\s*(.*)'
    candidate_re = r'C\d+:\s*(.*)'
    label_re = r'C(\d+)'
    
    all_messages = []
    ids = []

    if training:
        all_answers = []
        all_candidates = []
        all_candidates_ids = []

    # v8 prompt
    system_prompt = (
        "You serve as a personalized news recommendation system. Understand the User's History News, and then generate a recommendation list from the Candidate News ranked by how well they match the user's interests (i.e., similarity/continuity with the history)."
        "Output Format: Output your answer strictly in valid JSON with this exact structure and field order:\n\n"
        "{\n"
        '  "ranking": ["C#", "C#", ..., "C#"]\n'
        "}\n\n"
        "Important:\n"
        "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
        "- The ranking must be ordered from most relevant to least relevant.\n"
        "- Do not include anything outside the JSON block."
        )

    for row in df_data.iterrows():
        len_candidates = len(row[1]['candidate_news_id'].split('\n'))
        all_candidates_ids.append(row[1]['candidate_news_id'].split('\n'))
        user_prompt = (
            f"User's History News: {row[1]['history']}\n"
            f"Candidate News: {row[1]['candidate']}\n"
            f"Important: Imperative to include all {len_candidates} ids of the Candidate News set.\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]    
        # for gemma models
        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
        ]  

        if training:
            if len(row[1]['history_news_id'].split('\n')) <= 25:
                ids.append(row[1]['impression_id'])
                # build ground truth sequence, where the first elements are the relevant candidates and the rest are the other candidates
                ground_truths = row[1]['label'].split(",")
                all_candidates = ['C' + str(i) for i in range(1, len_candidates + 1)]
                # remove ground truth candidates from all candidates
                for gt in ground_truths:
                    all_candidates.remove(gt)
                # shuffle rest of candidates and append to ground truths
                random.shuffle(all_candidates)
                for c in all_candidates:
                    ground_truths.append(c)
                ground_truths_str = ' '.join(ground_truths)

                all_messages.append(messages)
                all_answers.append(ground_truths_str)
    return ids, all_messages, all_answers