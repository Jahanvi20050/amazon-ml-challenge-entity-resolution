# Multilingual Business Entity Resolution (Record Linkage)

This repository contains a high-precision, production-grade Multilingual Entity Resolution pipeline designed for hackathons and large-scale entity deduplication. The pipeline optimizes specifically for **Macro F_0.5** (precision-heavy metric penalizing false positive merges).

---

## 🌟 Key Architectural Features

1. **Multilingual NFKC Preprocessing (`preprocessing.py`)**:
   - Preserves non-ASCII Indic (Devanagari, Tamil, Kannada) and French scripts without stripping.
   - Cleans corporate suffix noise (`dba`, `LLC`, `Pvt. Ltd.`, `Inc`, `LLP`, etc.).
   - Normalizes transposed addresses and strips leading/trailing structural symbols.
   - Generates composite representations: `Name: {clean_name} | Address: {clean_address}`.

2. **Country-Partitioned FAISS Vector Blocking (`blocking.py`)**:
   - Dynamic partitioning by country (handles US, India, France, etc.).
   - Multilingual embeddings generated using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   - FAISS `IndexFlatIP` vector search retrieves top **35 candidates** per Source 1 entity.

3. **Multimodal Pairwise Feature Engineering (`feature_engineering.py`)**:
   - Vector cosine similarity.
   - RapidFuzz `token_set_ratio`, `partial_ratio`, `ratio`, and `token_sort_ratio` on names.
   - Address digit sequence Jaccard similarity (numerical token overlap).
   - Address character 3-gram Jaccard similarity.
   - Address fuzzy matching.

4. **Macro F_0.5 Calibrated Classifier (`train_and_predict.py`)**:
   - 80% train / 20% validation split on `Source 1`.
   - `LightGBMClassifier` trained on pair candidate features.
   - **Threshold Calibration**: Grid searches classification probability threshold on the 20% validation split specifically to maximize **Macro F_0.5** (strictly penalizing false positive merges and scoring singletons correctly as 1.0).
   - Test set inference and automated submission generation.

5. **Submission Validation (`utils/validate_submission.py`)**:
   - Checks tab-separated format rules for `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

---

## 🚀 Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
Run the full training, blocking, feature extraction, threshold calibration, inference, and validation pipeline:
```bash
python code/business_entity_resolution/src/train_and_predict.py
```

### 3. Validate Submission Outputs
To manually validate submission outputs at any time:
```bash
python utils/validate_submission.py
```

---

## 📁 Submission File Specifications

- **`output/candidate_pairs.tsv`**:
  - `source1_entity_id`: Entity ID from Source 1
  - `candidate_entity_ids`: Comma-separated list of top 35 candidate IDs retrieved via FAISS vector blocking

- **`output/matching_results.tsv`**:
  - `source1_entity_id`: Entity ID from Source 1
  - `matched_entity_ids`: Comma-separated list of matched entity IDs exceeding calibrated $F_{0.5}$ probability threshold (empty string for singletons)
