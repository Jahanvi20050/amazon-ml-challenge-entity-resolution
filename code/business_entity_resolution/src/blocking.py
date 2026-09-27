#!/usr/bin/env python3
"""
Enhanced Hybrid Blocking module for Multilingual Entity Resolution.
Combines:
1. Dense Vector Retrieval (SentenceTransformers + FAISS IndexFlatIP)
2. Sparse N-Gram & Word TF-IDF Similarity (Scikit-Learn TF-IDF Vectorizer)

Merges candidates from both dense and sparse space to achieve >99% blocking recall.
"""

import os
import csv
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class HybridBlocker:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"[Hybrid Blocking] Loading Dense SentenceTransformer model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)

    def encode_dense(self, texts: list, batch_size: int = 256) -> np.ndarray:
        """Encode texts into L2 normalized 384D dense vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def generate_candidate_pairs(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        top_k_dense: int = 30,
        top_k_sparse: int = 20
    ) -> dict:
        """
        Partition candidates and queries by country.
        Performs Hybrid Search (Dense FAISS IP + Sparse TF-IDF Cosine Sim).

        Returns:
            dict: s1_id -> list of dicts: {'cand_id': str, 'dense_sim': float, 'sparse_sim': float}
        """
        df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
        countries = set(df_s1['country'].unique()).union(set(df_cand['country'].unique()))

        results = {}

        for country in countries:
            sub_s1 = df_s1[df_s1['country'] == country].copy()
            sub_cand = df_cand[df_cand['country'] == country].copy()

            if sub_s1.empty:
                continue

            if len(sub_cand) == 0:
                print(f"[Blocking] Warning: Country '{country}' candidate pool is empty. Fallback to global pool.")
                sub_cand = df_cand.copy()

            s1_ids = sub_s1['entity_id'].tolist()
            s1_texts = sub_s1['composite_text'].tolist()

            cand_ids = sub_cand['entity_id'].tolist()
            cand_texts = sub_cand['composite_text'].tolist()

            print(f"[Blocking] Country '{country}': Queries={len(sub_s1)} | Candidates={len(sub_cand)}")

            # --- 1. Dense Vector Search (FAISS) ---
            cand_dense = self.encode_dense(cand_texts)
            s1_dense = self.encode_dense(s1_texts)

            dimension = cand_dense.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(cand_dense)

            k_dense = min(top_k_dense, len(cand_ids))
            dense_dists, dense_indices = index.search(s1_dense, k_dense)

            # --- 2. Sparse TF-IDF Search (Char 3-gram + Word) ---
            tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
            tfidf.fit(cand_texts + s1_texts)

            cand_sparse = tfidf.transform(cand_texts)
            s1_sparse = tfidf.transform(s1_texts)

            # Compute Sparse Dot Product (Cosine Similarity because rows are L2 normalized by TfidfVectorizer)
            sparse_sim_matrix = s1_sparse.dot(cand_sparse.T)

            # --- 3. Merge Hybrid Candidates per S1 entity ---
            k_sparse = min(top_k_sparse, len(cand_ids))

            for i, s1_id in enumerate(s1_ids):
                cand_map = {}

                # Add Dense Top Candidates
                for j in range(k_dense):
                    c_idx = dense_indices[i, j]
                    if 0 <= c_idx < len(cand_ids):
                        c_id = cand_ids[c_idx]
                        d_sim = float(dense_dists[i, j])
                        cand_map[c_id] = {'cand_id': c_id, 'dense_sim': d_sim, 'sparse_sim': 0.0}

                # Add Sparse Top Candidates
                row_sparse = sparse_sim_matrix[i].toarray().ravel()
                if len(row_sparse) > 0:
                    top_sparse_idxs = np.argpartition(row_sparse, -k_sparse)[-k_sparse:]
                    top_sparse_idxs = top_sparse_idxs[np.argsort(-row_sparse[top_sparse_idxs])]

                    for c_idx in top_sparse_idxs:
                        c_id = cand_ids[c_idx]
                        s_sim = float(row_sparse[c_idx])
                        if c_id in cand_map:
                            cand_map[c_id]['sparse_sim'] = s_sim
                        else:
                            # Dense sim fallback
                            cand_map[c_id] = {'cand_id': c_id, 'dense_sim': 0.0, 'sparse_sim': s_sim}

                results[s1_id] = list(cand_map.values())

        return results


def export_candidate_pairs_tsv(candidate_dict: dict, output_filepath: str):
    """Save candidate_pairs.tsv with columns [source1_entity_id, candidate_entity_ids]."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for s1_id, pair_list in candidate_dict.items():
            # Export top 35 candidates
            sorted_pairs = sorted(pair_list, key=lambda x: (x['dense_sim'] + x['sparse_sim']), reverse=True)[:35]
            cand_str = ",".join([c['cand_id'] for c in sorted_pairs])
            writer.writerow([s1_id, cand_str])
    print(f"[Blocking] Saved candidate pairs to {output_filepath}")
