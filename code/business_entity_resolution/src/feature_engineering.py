#!/usr/bin/env python3
"""
Enhanced Feature Engineering module for Multilingual Entity Resolution.
Extracts 16 high-discrimination similarity features:
1. Dense Vector Cosine Similarity
2. Sparse N-Gram TF-IDF Similarity
3. RapidFuzz Token Set Ratio, Partial Ratio, Ratio, Token Sort Ratio, WRatio
4. Jaro-Winkler Similarity on Clean Names
5. Address Numerical Token Overlap (Digit Sequence Jaccard)
6. Address Postal / Zip Code Exact Match Indicator
7. Address Character 3-Gram & 4-Gram Jaccard Similarities
8. Address Fuzzy Matching Ratios
9. Country Match Indicator & Relative Text Length Differences
"""

import re
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

try:
    from rapidfuzz.distance import JaroWinkler
    JARO_WINKLER_AVAIL = True
except ImportError:
    JARO_WINKLER_AVAIL = False


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
    dense_sim: float,
    sparse_sim: float
) -> dict:
    """
    Compute 16 pairwise similarity features for a single (S1, Candidate) pair.
    """
    s1_name = s1_row.get('clean_name', '')
    cand_name = cand_row.get('clean_name', '')
    
    s1_addr = s1_row.get('clean_address', '')
    cand_addr = cand_row.get('clean_address', '')

    s1_postal = s1_row.get('postal_code', '')
    cand_postal = cand_row.get('postal_code', '')

    # 1. RapidFuzz Name Features
    name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_wratio = fuzz.WRatio(s1_name, cand_name) / 100.0

    if JARO_WINKLER_AVAIL:
        name_jw = JaroWinkler.similarity(s1_name, cand_name)
    else:
        name_jw = name_ratio

    # 2. Address Digit Sequence Jaccard
    s1_digits = extract_digit_tokens(s1_addr)
    cand_digits = extract_digit_tokens(cand_addr)
    address_digit_jaccard = jaccard_similarity(s1_digits, cand_digits)

    # 3. Postal Code Match
    if s1_postal and cand_postal:
        postal_match = 1.0 if s1_postal == cand_postal else 0.0
    elif not s1_postal and not cand_postal:
        postal_match = 0.5
    else:
        postal_match = 0.0

    # 4. Address Character N-Gram Jaccards
    s1_3grams = extract_char_ngrams(s1_addr, n=3)
    cand_3grams = extract_char_ngrams(cand_addr, n=3)
    address_3gram_jaccard = jaccard_similarity(s1_3grams, cand_3grams)

    s1_4grams = extract_char_ngrams(s1_addr, n=4)
    cand_4grams = extract_char_ngrams(cand_addr, n=4)
    address_4gram_jaccard = jaccard_similarity(s1_4grams, cand_4grams)

    # 5. Address RapidFuzz Features
    address_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    address_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0

    # 6. Metadata & Relative Length Ratios
    same_country = 1.0 if s1_row.get('country') == cand_row.get('country') else 0.0

    max_n_len = max(len(s1_name), len(cand_name), 1)
    name_len_diff = abs(len(s1_name) - len(cand_name)) / max_n_len

    max_a_len = max(len(s1_addr), len(cand_addr), 1)
    address_len_diff = abs(len(s1_addr) - len(cand_addr)) / max_a_len

    return {
        'dense_cosine_sim': float(dense_sim),
        'sparse_tfidf_sim': float(sparse_sim),
        'name_token_set_ratio': name_token_set,
        'name_partial_ratio': name_partial,
        'name_ratio': name_ratio,
        'name_token_sort_ratio': name_token_sort,
        'name_wratio': name_wratio,
        'name_jaro_winkler': name_jw,
        'address_digit_jaccard': address_digit_jaccard,
        'postal_code_match': postal_match,
        'address_3gram_jaccard': address_3gram_jaccard,
        'address_4gram_jaccard': address_4gram_jaccard,
        'address_token_set_ratio': address_token_set,
        'address_partial_ratio': address_partial,
        'same_country': same_country,
        'name_len_diff': name_len_diff,
        'address_len_diff': address_len_diff,
    }


def build_feature_matrix(
    s1_dict: dict,
    cand_dict: dict,
    candidate_pairs: dict,
    ground_truth_dict: dict = None
) -> tuple:
    """
    Construct feature DataFrame and label array from hybrid candidate pairs.

    Returns:
        (df_features, pair_info_df, y_labels)
    """
    rows = []
    pair_info = []
    labels = []

    for s1_id, pairs in candidate_pairs.items():
        s1_row = s1_dict.get(s1_id, {})
        true_matches = ground_truth_dict.get(s1_id, set()) if ground_truth_dict else set()

        for c_info in pairs:
            cand_id = c_info['cand_id']
            dense_sim = c_info.get('dense_sim', 0.0)
            sparse_sim = c_info.get('sparse_sim', 0.0)

            cand_row = cand_dict.get(cand_id, {})
            feats = compute_pair_features(s1_row, cand_row, dense_sim, sparse_sim)
            rows.append(feats)
            pair_info.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id})

            if ground_truth_dict is not None:
                labels.append(1 if cand_id in true_matches else 0)

    df_features = pd.DataFrame(rows)
    pair_info_df = pd.DataFrame(pair_info)
    y_labels = np.array(labels) if ground_truth_dict is not None else None

    return df_features, pair_info_df, y_labels
