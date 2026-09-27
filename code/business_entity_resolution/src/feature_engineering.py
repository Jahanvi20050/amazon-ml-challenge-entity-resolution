#!/usr/bin/env python3
"""
Feature Engineering module for Multilingual Entity Resolution.
Extracts pairwise matching features between Source 1 records and Candidate pool records:
- Embedding Cosine Similarity
- RapidFuzz token_set_ratio & partial_ratio on clean business names
- RapidFuzz ratio & token_sort_ratio
- Address digit sequence Jaccard similarity (numerical token overlap)
- Address character 3-gram Jaccard similarity
- Address token_set_ratio & partial_ratio
"""

import re
import pandas as pd
import numpy as np
from rapidfuzz import fuzz


def extract_digit_tokens(text: str) -> set:
    """Extract set of digit sequence strings from text."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\d+', text))


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0


def extract_char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    if not isinstance(text, str) or len(text) < n:
        return set([text]) if isinstance(text, str) and text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def compute_pair_features(
    s1_row: dict,
    cand_row: dict,
    vec_sim: float
) -> dict:
    """
    Compute comprehensive pairwise feature dictionary for a single (S1, Candidate) pair.
    """
    s1_name = s1_row.get('clean_name', '')
    cand_name = cand_row.get('clean_name', '')
    
    s1_addr = s1_row.get('clean_address', '')
    cand_addr = cand_row.get('clean_address', '')

    # RapidFuzz name features
    name_token_set_ratio = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    name_token_sort_ratio = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0

    # Address numerical token overlap (Jaccard similarity on digit sequences)
    s1_digits = extract_digit_tokens(s1_addr)
    cand_digits = extract_digit_tokens(cand_addr)
    address_digit_jaccard = jaccard_similarity(s1_digits, cand_digits)

    # Address character 3-gram Jaccard similarity
    s1_3grams = extract_char_ngrams(s1_addr, n=3)
    cand_3grams = extract_char_ngrams(cand_addr, n=3)
    address_char_3gram_jaccard = jaccard_similarity(s1_3grams, cand_3grams)

    # RapidFuzz address features
    address_token_set_ratio = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    address_partial_ratio = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0

    # Country match
    same_country = 1.0 if s1_row.get('country') == cand_row.get('country') else 0.0

    return {
        'vector_cosine_sim': float(vec_sim),
        'name_token_set_ratio': name_token_set_ratio,
        'name_partial_ratio': name_partial_ratio,
        'name_ratio': name_ratio,
        'name_token_sort_ratio': name_token_sort_ratio,
        'address_digit_jaccard': address_digit_jaccard,
        'address_char_3gram_jaccard': address_char_3gram_jaccard,
        'address_token_set_ratio': address_token_set_ratio,
        'address_partial_ratio': address_partial_ratio,
        'same_country': same_country,
    }


def build_feature_matrix(
    s1_dict: dict,
    cand_dict: dict,
    candidate_pairs: dict,
    ground_truth_dict: dict = None
) -> tuple:
    """
    Construct feature DataFrame and label array from candidate pairs.
    
    Args:
        s1_dict: dict mapping s1_id -> row dict/Series
        cand_dict: dict mapping cand_id -> row dict/Series
        candidate_pairs: dict mapping s1_id -> list of (cand_id, cosine_sim)
        ground_truth_dict: dict mapping s1_id -> set of true matched cand_ids (optional for train)

    Returns:
        (df_features, pair_info_df, labels)
    """
    rows = []
    pair_info = []
    labels = []

    for s1_id, pairs in candidate_pairs.items():
        s1_row = s1_dict.get(s1_id, {})
        true_matches = ground_truth_dict.get(s1_id, set()) if ground_truth_dict else set()

        for cand_id, vec_sim in pairs:
            cand_row = cand_dict.get(cand_id, {})
            feats = compute_pair_features(s1_row, cand_row, vec_sim)
            rows.append(feats)
            pair_info.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id})

            if ground_truth_dict is not None:
                labels.append(1 if cand_id in true_matches else 0)

    df_features = pd.DataFrame(rows)
    pair_info_df = pd.DataFrame(pair_info)
    y_labels = np.array(labels) if ground_truth_dict is not None else None

    return df_features, pair_info_df, y_labels
