# Ditto ER Pipeline (Config-Driven, No Notebook)

This pipeline builds **gold tables**, creates **blocking-based candidate pairs**, generates
`train/valid/test/predict` Ditto text files, trains a Ditto model, and writes test/predict reports.

---

## 0) Setup: repository and dependencies

### Get the code

Clone or copy this **Ditto** project so your working tree includes at least:

- `pipeline/` — CLI stages (`run_pipeline`, `build_gold`, …)
- `er_pipeline/` — blocking helpers (SBERT, TF-IDF, ANN, n-gram)
- `configs/` — dataset, blocking, and training YAML
- `FAIR-DA4ER/` — upstream **FAIR-DA4ER** repo with `FAIR-DA4ER/ditto/ditto_light` (used for Ditto model + tokenizers)

Example:

```bash
git clone <YOUR_REPOSITORY_URL> Ditto
cd Ditto
```

If **FAIR-DA4ER** is not already inside `Ditto/`, add it (submodule, second clone, or vendor copy) so this path exists:

`FAIR-DA4ER/ditto/ditto_light/`

For example:

```bash
cd Ditto
git clone https://github.com/MarcoNapoleone/FAIR-DA4ER.git FAIR-DA4ER
```

(Use the URL that matches how you maintain **FAIR-DA4ER** in your project.)

### Python environment

Use **Python 3.10+** (3.12 is fine). A virtual environment is recommended:

```bash
cd Ditto
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
pip install --upgrade pip
```

### Install dependencies

From the **Ditto** root (the directory that contains `requirements.txt`):

```bash
pip install -r requirements.txt
```

This file bundles:

- **Pipeline**: `pandas`, `numpy`, `pyyaml`, `scikit-learn`, `scipy`
- **Training / SBERT / transformers**: `torch`, `transformers`, `sentence-transformers`, `tqdm`
- **ANN blocking** (`strategy: ann`): `faiss-cpu` (use `faiss-gpu` instead if you rely on GPU FAISS and know your CUDA stack)
- **FAIR-DA4ER / Ditto stack** (aligned with `FAIR-DA4ER/requirements.txt`): `gensim`, `spacy`, `nltk`, etc.
- **Optional analysis plots**: `matplotlib`

If you only need a minimal subset (e.g. TF-IDF + n-gram blocking and no `spacy`), you can install packages selectively; the full file matches what this repo is built and tested with.

### Optional: Hugging Face Hub

For higher rate limits when downloading models, set:

```bash
export HF_TOKEN=<your_token>
```

### Optional: spaCy English model

Some FAIR-DA4ER / Ditto paths expect spaCy; if you hit import or model errors:

```bash
python -m spacy download en_core_web_sm
```

### Where to run commands

All pipeline examples below assume the **current working directory** is the **Ditto** root (where `pipeline/` and `configs/` live).

---

## 1) Directory Contract

For each dataset, use:

```
dataset/
  <dataset>/
    <dataset>.csv
    standardized_<dataset>.csv
    gold_<dataset>.csv           # generated or reused
    predict_<dataset>.csv        # input for prediction pairing
```

Pipeline outputs:

```
data/
  <dataset>/
    <dataset>_train.txt
    <dataset>_valid.txt
    <dataset>_test.txt
    <dataset>_predict.txt

models/
  <dataset>/
    best_model.pt
    train_metrics.json

Test_Output/
  <dataset>_test_result/
    <dataset>_test.csv
    <dataset>_test_analysis.txt

predict_output/
  <dataset>_predict_result/
    <dataset>_predict.csv
    <dataset>_predict_analysis.txt
```

---

## 2) Config Files

- Dataset config: `configs/datasets/<dataset>.yaml`
- Blocking config: `configs/blocking.yaml`
- Training config: `configs/training.yaml`

Default example included:
- `configs/datasets/pro_supplier.yaml`

---

## 3) Run Commands

From repository root (`/workspace/Ditto`):

### Full end-to-end run

```bash
python -m pipeline.run_pipeline --dataset pro_supplier --stage all
```

### Stage-by-stage

```bash
python -m pipeline.run_pipeline --dataset pro_supplier --stage build_gold
python -m pipeline.run_pipeline --dataset pro_supplier --stage build_splits
python -m pipeline.run_pipeline --dataset pro_supplier --stage build_predict
python -m pipeline.run_pipeline --dataset pro_supplier --stage train
python -m pipeline.run_pipeline --dataset pro_supplier --stage test
python -m pipeline.run_pipeline --dataset pro_supplier --stage predict
```

You can also call individual modules directly:

```bash
python -m pipeline.build_gold --dataset pro_supplier
python -m pipeline.build_train_valid_test --dataset pro_supplier
python -m pipeline.build_predict_pairs --dataset pro_supplier
python -m pipeline.train_model --dataset pro_supplier
python -m pipeline.test_and_analyze --dataset pro_supplier
python -m pipeline.predict_and_analyze --dataset pro_supplier
```

---

## 4) Blocking Strategy Parameters

In `configs/blocking.yaml`:

- `strategy`: `sbert` | `ann` | `ngram` | `tfidf`
- `top_k_train`: top-k candidates per anchor for train split generation
- `target_total_pairs`: approximate total pair count goal
- `top_k_predict`: top-k gold candidates for each predict record (default `1000`)

For `sbert` strategy:
- `model_name`, `batch_size_encode`, `block_rows`, `use_gpu`

For `ann` strategy:
- `nlist`, `nprobe`

For `ngram` strategy:
- `n` (character n-gram size)

For `tfidf` strategy:
- `max_features`, `min_df`, `max_df`, `sublinear_tf` (vectorizer)
- `anchor_batch_size`, `block_rows` (memory vs speed for similarity blocks; CPU-only)

---

## 5) Notes

- `build_gold` will use existing `canonical_int` if already present in raw CSV,
  otherwise it joins standardized table to raw table using configured join keys.
- Ditto text rows are generated in standard format:
  `record_left<TAB>record_right<TAB>label`
- Predict file uses dummy label (`0`) and model inference generates actual predictions.

---

## 6) Quick Troubleshooting

- If training fails to import Ditto classes, verify this path exists:
  `FAIR-DA4ER/ditto/ditto_light`
- If no pairs are generated, reduce blocking strictness or increase `top_k_train`.
- If prediction output is too large, reduce `top_k_predict`.
