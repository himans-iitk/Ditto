## IJF Supplier Matching with Sentence-BERT Blocking, ANN, and Ditto

This document describes the complete pipeline implemented in `ditto_entity_resolution.ipynb` for scalable supplier matching on the IJF dataset using **Sentence-BERT** for blocking, **approximate nearest neighbours (ANN)** for efficiency, and **Ditto** for supervised entity resolution. It also explains the main design choices (including clustering by `canonical_int`) and how to interpret the resulting metrics.

---

## 1. Problem and Constraints

- **Goal**: Train a Ditto-style ER model to decide whether two supplier records refer to the same entity.
- **Data**: Large IJF supplier table:
  - File: `Ditto/FAIR-DA4ER/ditto/data/ijf/blocking_sentence_bert/pro_supplier_with_clean_and_canonical_trimmed.csv`
  - ~1.39M rows, each with:
    - `clean_supplier_name`
    - Address/location fields
    - `canonical_int` (integer cluster ID for the supplier)
- **Constraints**:
  - Blocking must be feasible at this scale (no O(n²) all-pairs).
  - Ditto training + evaluation must complete in **≈20–30 minutes** on a single GPU (RTX 5090).

---

## 2. Core Representations and Labels

### 2.1 Ditto Record Serialization

Each CSV row is converted to a Ditto-style serialized record:

- Example pattern:
  - `COL clean_supplier_name VAL ... COL address VAL ... COL city VAL ... COL prov VAL ... COL postal VAL ... COL country VAL ...`
- These serialized records are stored as:
  - `records_blocking = [(row_index_as_str, serialized_record_str), ...]`

### 2.2 Canonical Clusters via `canonical_int`

- The CSV contains a **`canonical_int`** column, which serves as a cluster ID:
  - All records with the same non-zero `canonical_int` are treated as the **same supplier**.
  - `canonical_int == 0` is treated as “unclustered / unknown”.

We use `canonical_int` to:

1. **Define ground-truth labels** for pairwise ER:
   - A pair `(i, j)` is labeled:
     - `1` (match) if `canonical_ints[i] == canonical_ints[j] != 0`
     - `0` (non-match) otherwise.
2. **Drive cluster-level splitting**:
   - Instead of random pair-level splits, we split at the **cluster** level by `canonical_int`.

### 2.3 Cluster Statistics

The notebook computes:

- For each `canonical_int`:
  - Count of records in the cluster.
  - An example `clean_supplier_name` extracted via regex from the Ditto record.
- Overall distribution:
  - Number of distinct `canonical_int` values (including 0).
  - Min / max / mean / median **records per canonical cluster**.
  - Top 20 clusters by size, each with a representative cleaned name.

This helps understand:

- How many very large (“head”) suppliers exist.
- How many small (“tail”) clusters exist.

---

## 3. Sentence-BERT Embeddings

We use **Sentence-BERT (MiniLM)** as the base representation for blocking:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Device: `cuda` if available, else `cpu`.
- Input: Ditto record string for each row.
- Output:
  - A matrix `V ∈ ℝ^{N×d}` of normalized embeddings (`np.float32`).
  - For ANN, this is cached as `emb_ann` to avoid re-encoding.

Normalization (L2) ensures that:

- Inner product ≈ cosine similarity.
- Useful for both dense and ANN-based nearest neighbour search.

---

## 4. Blocking Strategies

The pipeline uses two related blocking strategies based on Sentence-BERT:

1. **Dense top‑K blocking with random row sampling** (BERT-blocking).
2. **ANN-based blocking with FAISS** (ANN-blocking).

Both aim to generate a **manageable candidate set** of record pairs to label and feed into Ditto, instead of considering all O(n²) possibilities.

### 4.1 BERT-Blocking (Dense with Sampling)

Initial attempt:

- Compute `V @ V.T` in batches and, for each row, take top‑K most similar neighbours.
- For N ≈ 1.38M and K=20:
  - This can produce tens of millions of pairs.
  - Train files can easily reach tens of GB.

To reduce cost and file size:

- Use **random row sampling**:
  - Randomly choose a subset of rows.
  - For each sampled row:
    - Compute dense similarities against all records with `batch_vecs @ V.T`.
    - Take top‑K neighbours (excluding self).
  - Deduplicate candidate pairs (treat `(i, j)` and `(j, i)` as one).
  - Cap total candidate count to a target (e.g. 100k).

This yields:

- A moderate number of candidate pairs (e.g. 100k–200k).
- Reasonable disk usage and Ditto training time.

### 4.2 ANN-Blocking with FAISS

For more scalable blocking, we build a **FAISS IVF-Flat** index:

- Training and index construction:
  - Use `IndexIVFFlat` with `METRIC_INNER_PRODUCT`.
  - Parameters:
    - `nlist = 4096` coarse clusters.
    - `nprobe = 16` clusters probed per search.
  - Train on up to 200k randomly sampled vectors.
  - Add all embeddings `V_ann` to the index.

- Candidate generation:
  - Choose:
    - `TOP_K_ANN` = e.g. 200 neighbours per query.
    - `TARGET_TOTAL_PAIRS_ANN` = e.g. 200,000 pairs.
  - Compute number of queries: `n_queries = TARGET_TOTAL_PAIRS_ANN // TOP_K_ANN`.
  - Randomly pick `n_queries` distinct rows as query points.
  - For each, search `index_ivf` for top‑K neighbours:
    - Exclude self-matches.
  - Build `(i, j)` pairs, deduplicate, and subsample back down to `TARGET_TOTAL_PAIRS_ANN` if needed.

Advantages:

- More scalable for larger N or larger K than dense search.
- Tunable trade-offs via `nlist`, `nprobe`, K, and `TARGET_TOTAL_PAIRS_ANN`.

---

## 5. Labeling and Initial Pair-Level Splits

For any candidate list (BERT or ANN), we:

1. **Label pairs** using `canonical_int`:
   - Positive (1): same non-zero `canonical_int`.
   - Negative (0): otherwise.

2. **Pair-level splits (initially)**:
   - Stratified 80% train, 10% valid, 10% test by label.

This gives a quick training dataset but has two downsides:

- The same `canonical_int` cluster can appear across train/valid/test.
- Validation and test performance may be **over-optimistic** because the model has seen the same suppliers during training.

---

## 6. Distribution Analysis and Motivation for Cluster-Level Splits

To better understand the dataset structure, we compute:

1. For each split (train/valid/test) and each blocking method (BERT, ANN):
   - For each record:
     - Number of **match pairs** (positive label) it participates in.
     - Number of **non-match pairs** it participates in.
   - Distributions show:
     - Some records (large suppliers) appear in many pairs.
     - Many records appear in few pairs.

2. For `canonical_int`:
   - Histogram of **cluster sizes** (records per canonical_int).
   - Heavy-tailed distribution: a few very large clusters and many tiny ones.

These observations motivated the shift to **cluster-level splits**:

- We want validation and test to contain **entirely new canonical clusters**, not just unseen pairs from clusters the model has already seen.

---

## 7. Cluster-Level Splitting by `canonical_int`

### 7.1 Defining Cluster Splits

We partition canonical clusters rather than individual pairs:

1. Collect distinct `canonical_int` values (including 0).
2. Shuffle them with a fixed seed.
3. Split into:
   - `canon_train` (80% of clusters),
   - `canon_valid` (10%),
   - `canon_test`  (10%),
   ensuring they are disjoint.

### 7.2 Cluster-Level BERT-Blocking Datasets

Given BERT-blocking pairs `labeled_pairs` with fields `(i, j, label)`:

1. For each pair:
   - Let `ci = canonical_ints[i]`, `cj = canonical_ints[j]`.
2. Assign the pair to a split **only if both clusters** belong to the same cluster split:
   - If `ci ∈ canon_train` and `cj ∈ canon_train` → train.
   - If `ci ∈ canon_valid` and `cj ∈ canon_valid` → valid.
   - If `ci ∈ canon_test`  and `cj ∈ canon_test`  → test.
3. Write three files under:
   - `data/ijf/blocking_sentence_bert_cluster/`:
     - `train.txt`, `valid.txt`, `test.txt`.
   - Each line: `rec1_str \t rec2_str \t label_int`.

The notebook prints:

- Pair counts and label distributions for Train/Valid/Test.

### 7.3 Cluster-Level ANN-Blocking Datasets

Repeat the same process for ANN-blocking pairs `labeled_pairs_ann`:

1. Use `canonical_ints` to filter pairs into train/valid/test based on `canon_train`, `canon_valid`, `canon_test`.
2. Write three files under:
   - `data/ijf/ANN_cluster/`:
     - `train.txt`, `valid.txt`, `test.txt`.
3. Print per-split statistics (pair counts, % positives vs % negatives).

**Key effect**:

- Train/valid/test datasets contain **disjoint sets of suppliers** (canonical_int clusters).
- Test performance now reflects the model’s ability to handle **new clusters**, not just new pairs of known clusters.

---

## 8. Ditto Model and Training Configuration

### 8.1 Model

- LM backbone: DistilBERT (`distilbert-base-uncased`).
- Architecture:
  - `[CLS]` vector from DistilBERT.
  - Linear layer from hidden size to 2 logits (no hidden MLP).

### 8.2 Datasets

For each dataset variant (BERT-blocking, ANN-blocking, and their cluster-level versions):

- Use `DittoDataset` with:
  - `max_len = MAX_LEN` (typically 256).
  - Tokenizer from `transformers` corresponding to the LM.

### 8.3 Training Loop

- Epochs: `N_EPOCHS = 5`.
- Optimizer: `AdamW`.
- Learning rate: `LR = 3e-5`.
- Scheduler: `get_linear_schedule_with_warmup` with no warmup.
- Loss: `CrossEntropyLoss`.
- Device: `cuda` if available, else `cpu`.
- DataLoader settings:
  - Train: `batch_size ≈ 64`, `num_workers=4`, `pin_memory=True`, `persistent_workers=True`.
  - Eval:  `batch_size ≈ 128`, `num_workers=2`, `pin_memory=True`.

For each epoch:

1. Train on train loader (`train_epoch`):
   - Standard forward → loss → backward → optimizer.step → scheduler.step.
2. Evaluate on valid and test loaders (`evaluate`):
   - Compute probability of class 1 for each pair.
   - For validation:
     - Search over thresholds in `[0.0, 1.0)` in steps of 0.05 to find threshold with maximum F1.
   - For test:
     - Apply validation-derived threshold (`dev_th`) to probabilities.
     - Compute test F1 under this fixed threshold.
3. Track best validation F1 and corresponding test F1.

### 8.4 Final Evaluation

After training completes:

- Run a final evaluation on the test set with the best threshold from the validation set.
- Use `classification_report` from `sklearn` to print:
  - Precision, recall, F1 for each class (No match / Match).
  - Macro and weighted averages.
- Compute:
  - TP, FP, TN, FN.
  - Accuracy, Precision, Recall, F1.

This is done for:

1. BERT-blocking datasets.
2. ANN-blocking datasets.
3. BERT cluster-level datasets.
4. ANN cluster-level datasets.

---

## 9. Empirical Results (Qualitative Summary)

Exact numbers are in the notebook outputs, but overall trends are:

- **ANN-blocking (pair-level split)**:
  - Very high performance:
    - F1 ≈ 0.9995.
    - Only a handful of FP and FN in ~20k test pairs.
  - Interpretation:
    - Canonical clusters are coherent.
    - Sentence-BERT embeddings + ANN blocking provide strong features.
    - However, train/test share clusters, so results may overestimate generalization.

- **BERT-blocking and ANN-blocking (cluster-level splits)**:
  - Performance remains very strong after moving to cluster-level splits.
  - Ditto generalizes well to entirely new clusters.
  - This is a more realistic evaluation of how the model will behave on unseen suppliers.

In all cases, you maintain:

- **Training time within ≈20–30 minutes** on RTX 5090.
- **Dataset sizes in the hundreds of thousands of pairs**, which is manageable.

---

## 10. Design Choices and Rationale

1. **Sentence-BERT (MiniLM)**:
   - Strong semantic representation for record similarity.
   - Lightweight enough to encode 1.4M records in a couple of minutes on GPU.

2. **Random sampling + top‑K for dense blocking**:
   - Avoids O(n²) explosion.
   - Easy to bound candidate counts with `TARGET_TOTAL_PAIRS`.

3. **ANN blocking with FAISS (IVF-Flat)**:
   - Enables larger K and/or more queries at scale.
   - Tunable accuracy-speed tradeoff via `nlist`, `nprobe`.

4. **Labeling by `canonical_int`**:
   - Scalable way to derive millions of labeled pairs.
   - Simplifies evaluation, as canonical labels are treated as ground truth.

5. **Cluster-level split by `canonical_int`**:
   - Prevents leakage of supplier clusters across train/valid/test.
   - Gives more realistic generalization metrics: model must handle unseen clusters.

6. **Optimized Ditto training**:
   - Larger batch sizes and DataLoader optimizations keep training time low.
   - 5-epoch schedule balances convergence with runtime.

Taken together, these decisions give you a practical and robust ER pipeline that:

- Scales to a **very large** supplier table.
- Trains within **20–30 minutes**.
- Provides high-fidelity estimates of model performance under realistic deployment scenarios.

---

## 11. How to Reproduce (High-Level)

Inside `ditto_entity_resolution.ipynb`, run in order:

1. **Setup and base Ditto training** (Part A of the notebook).
2. **IJF CSV load and Ditto record serialization**.
3. **Sentence-BERT encoding** (`records_blocking` → embeddings).
4. **BERT-blocking**:
   - Random-sampled dense top‑K pairs → `labeled_pairs`.
   - Label and (optionally) split/write BERT-blocking datasets.
5. **ANN-blocking with FAISS**:
   - Build IVF index over embeddings.
   - Generate top‑K ANN candidate pairs → `labeled_pairs_ann`.
   - Label and split/write ANN-blocking datasets.
6. **Distribution & cluster statistics**:
   - Per-record match/non-match counts.
   - Canonical_int cluster size distributions and examples.
7. **Cluster-level splitting (Steps C1–C7)**:
   - Define `canon_train`, `canon_valid`, `canon_test`.
   - Build/write BERT and ANN cluster-level datasets.
   - Train Ditto on each and evaluate final performance and confusion matrices.

Refer to the notebook for exact cell numbers and outputs; this README focuses on the conceptual and design-level overview.

