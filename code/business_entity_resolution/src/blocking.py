#!/usr/bin/env python3
"""
Blocking module for Multilingual Entity Resolution.
Partitions records by country, computes embeddings using paraphrase-multilingual-MiniLM-L12-v2,
and uses FAISS IndexFlatIP to retrieve top 35 candidate pairs per S1 entity.
"""

import os
import csv
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors for cosine similarity computation via Inner Product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return (vectors / norms).astype(np.float32)


class SentenceTransformerBlocker:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"[Blocking] Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list, batch_size: int = 256) -> np.ndarray:
        """Encode a list of text strings into normalized L2 embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def generate_candidate_pairs(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        top_k: int = 35
    ) -> dict:
        """
        Partition by country, index candidate pool (S2 + S3) with FAISS IndexFlatIP,
        and retrieve top_k candidates for each S1 record.

        Returns:
            dict: mapping source1_entity_id -> list of (candidate_entity_id, cosine_sim)
        """
        # Combine S2 and S3 as candidates
        df_cand = pd.concat([df_s2, df_s3], ignore_index=True)

        # Unique countries across queries and candidates
        countries = set(df_s1['country'].unique()).union(set(df_cand['country'].unique()))

        results = {}

        # Handle candidate retrieval per country partition
        for country in countries:
            sub_s1 = df_s1[df_s1['country'] == country].copy()
            sub_cand = df_cand[df_cand['country'] == country].copy()

            if sub_s1.empty:
                continue

            print(f"[Blocking] Processing Country: '{country}' | S1 query count: {len(sub_s1)} | Candidate pool (S2+S3) count: {len(sub_cand)}")

            # Fallback to full candidate pool if country candidate pool is empty or too small
            if len(sub_cand) == 0:
                print(f"[Blocking] Warning: No candidates found for country '{country}'. Falling back to global candidate pool.")
                sub_cand = df_cand.copy()

            # Compute embeddings for sub_cand
            cand_texts = sub_cand['composite_text'].tolist()
            cand_ids = sub_cand['entity_id'].tolist()
            cand_embeddings = self.encode_texts(cand_texts)

            # Compute embeddings for sub_s1
            s1_texts = sub_s1['composite_text'].tolist()
            s1_ids = sub_s1['entity_id'].tolist()
            s1_embeddings = self.encode_texts(s1_texts)

            # Build FAISS IndexFlatIP
            dimension = cand_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(cand_embeddings)

            # Search top_k
            actual_k = min(top_k, len(cand_ids))
            distances, indices = index.search(s1_embeddings, actual_k)

            # Record matches for each query
            for i, s1_id in enumerate(s1_ids):
                pair_list = []
                for j in range(actual_k):
                    cand_idx = indices[i, j]
                    sim = float(distances[i, j])
                    if cand_idx >= 0 and cand_idx < len(cand_ids):
                        cand_id = cand_ids[cand_idx]
                        pair_list.append((cand_id, sim))
                results[s1_id] = pair_list

        return results


def export_candidate_pairs_tsv(candidate_dict: dict, output_filepath: str):
    """Save candidate_pairs.tsv with columns [source1_entity_id, candidate_entity_ids]."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for s1_id, pair_list in candidate_dict.items():
            cand_str = ",".join([c[0] for c in pair_list])
            writer.writerow([s1_id, cand_str])
    print(f"[Blocking] Exported candidate pairs to {output_filepath}")
