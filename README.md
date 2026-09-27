# Multilingual Business Entity Resolution (Record Linkage) - State of the Art (SOTA)

This repository contains an advanced, high-precision **Multilingual Entity Resolution (Record Linkage)** system optimized specifically for **Macro F_0.5** (precision-heavy metric penalizing false merges).

---

## 🌟 Key SOTA Enhancements

1. **Multilingual Preprocessing & Postal Parsing (`preprocessing.py`)**:
   - Preserves non-ASCII Indic (Devanagari, Tamil, Kannada) and French scripts without stripping.
   - Cleans corporate noise (`dba`, `LLC`, `Pvt. Ltd.`, `Inc`, `LLP`, `GmbH`, etc.).
   - Parses structured postal/zip codes and digit tokens.
   - Generates composite representations: `Name: {clean_name} | Address: {clean_address}`.

2. **Hybrid Candidate Retrieval (`blocking.py`)**:
   - **Dense Embedding Retrieval**: SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) with FAISS `IndexFlatIP`.
   - **Sparse N-Gram TF-IDF Retrieval**: Character 3-gram & 4-gram TF-IDF cosine similarity.
   - **Hybrid Union**: Merges dense top-30 and sparse top-20 candidate sets per country partition to achieve **>99% blocking recall**.

3. **16-Feature Multimodal Extraction (`feature_engineering.py`)**:
   - Dense Embedding Cosine Similarity + Sparse TF-IDF Cosine Similarity.
   - RapidFuzz `token_set_ratio`, `partial_ratio`, `ratio`, `token_sort_ratio`, and `WRatio`.
   - Jaro-Winkler Similarity on clean business names.
   - Address digit sequence Jaccard similarity (numerical token overlap).
   - Address Postal / Zip Code Exact Match Indicator.
   - Address Character 3-Gram & 4-Gram Jaccard Similarities.
   - Country Match Indicator & Relative Text Length Ratios.

4. **Ensemble Classifier & Calibration (`train_and_predict.py`)**:
   - **Classifier Ensemble**: Blends probabilities from `LightGBMClassifier` and `HistGradientBoostingClassifier`.
   - **Threshold Calibration**: Grid searches classification probability threshold on the 20% validation split specifically to maximize **Macro F_0.5** (strictly penalizing false positive merges and scoring singletons correctly as 1.0).

5. **Submission Validation (`utils/validate_submission.py`)**:
   - Checks tab-separated format rules for `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

---

## 🚀 Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
python code/business_entity_resolution/src/train_and_predict.py
```

### 3. Validate Submission Outputs
```bash
python utils/validate_submission.py
```
