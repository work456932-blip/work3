# CIRCLE: Anonymous Code Supplement

This package contains an anonymous implementation of the CIRCLE pipeline for
patient-disjoint multimodal clinical prediction. It includes structured
evidence construction, retrieval, conflict localization, bounded repair,
probability calibration, conformal intervals, and selective abstention.

## Data and privacy

No clinical images, reports, patient identifiers, model checkpoints, API
credentials, or generated experiment outputs are included. Obtain MIMIC-CXR,
CheXpert, and OpenI from their official distribution channels and comply with
the corresponding licenses and data-use requirements. Keep all subject-level
splits disjoint and construct retrieval indexes from training subjects only.

## Quick start

```bash
python -m pip install -r requirements.txt
python sanity_check.py --config configs/default.yaml
python run_pipeline.py --config configs/default.yaml --study-id demo-study-001
```

`sanity_check.py` creates synthetic placeholder inputs under `./artifacts` so
that the software path can be tested without distributing clinical data.

## Main scripts

- `build_auto_labels.py`: construct task-label tables from locally available inputs
- `fit_conformal.py`: fit a split-conformal residual quantile
- `run_pipeline.py`: run the complete pipeline for one study
- `run_batch.py`: run the pipeline over a study manifest
- `run_offline.py`: run without external API-backed tools
- `train_contract_model.py`: fit a lightweight contract model from traces
- `train_repair_policy.py`: fit an optional repair policy from traces
- `train_vision_heads.py`: fit lightweight vision-head statistics

## Expected local layout

```text
artifacts/
  index/
    meta.pkl
    image.npy
    text.npy
  labels.csv
  studies.csv
  notes/
    <study_id>.txt
  reports/
    <study_id>.txt
  images/
    <study_id>_0.png
```

Paths and model settings can be changed in `configs/default.yaml`. Secrets
should be provided through environment variables or a local untracked config;
do not place credentials in the repository.
