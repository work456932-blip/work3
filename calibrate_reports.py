from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASKS = ["mortality_risk", "icu_risk", "ventilation_risk"]
BASE_MODELS = ["lr", "rf", "et", "gb", "gb3", "hgb", "hgb_deep"]


CONCEPT_PATTERNS = {
    "opacity": r"\b(opacity|opacities|airspace)\b",
    "consolidation": r"\bconsolidation\b",
    "infiltrate": r"\b(infiltrate|infiltrates|infiltration)\b",
    "edema": r"\b(edema|oedema|pulmonary edema|interstitial edema)\b",
    "effusion": r"\b(pleural effusion|effusions?|blunting)\b",
    "pneumothorax": r"\bpneumothorax\b",
    "atelectasis": r"\batelectasis\b",
    "pneumonia": r"\bpneumonia\b",
    "bilateral": r"\b(bilateral|diffuse|multifocal|bibasilar)\b",
    "low_volume": r"\b(low lung volume|low lung volumes|hypoinflation|hypoventilation)\b",
    "tube": r"\b(tube|ett|endotracheal|tracheostomy|trach)\b",
    "ventilator": r"\b(ventilator|mechanical ventilation|ventilated|intubat|intubation)\b",
    "cpap_bipap": r"\b(cpap|bipap|noninvasive ventilation)\b",
    "line_device": r"\b(line|catheter|central venous|picc|pacer|device)\b",
    "resp_failure": r"\b(respiratory failure|respiratory distress|ards)\b",
    "hypoxia": r"\b(hypoxia|hypoxemia|hypoxic)\b",
    "shock": r"\b(shock|septic shock|cardiogenic shock)\b",
    "sepsis": r"\b(sepsis|septic)\b",
    "arrest": r"\b(arrest|code blue|cardiac arrest)\b",
    "icu": r"\b(icu|micu|sicu|ccu|critical care)\b",
    "severe": r"\b(severe|critical|unstable|decompensation)\b",
    "cardiomegaly": r"\b(cardiomegaly|enlarged heart|heart is enlarged)\b",
    "vascular_congestion": r"\b(vascular congestion|pulmonary vascular|vascular engorgement|congestion)\b",
    "heart_failure": r"\b(heart failure|congestive heart failure|chf)\b",
    "tachycardia": r"\btachycardia\b",
    "hypotension": r"\b(hypotension|hypotensive|vasopressor)\b",
    "clear_lungs": r"\b(lungs are clear|clear lungs|no acute cardiopulmonary|no focal airspace)\b",
    "no_effusion": r"\b(no pleural effusion|without pleural effusion)\b",
    "no_pneumothorax": r"\b(no pneumothorax|without pneumothorax)\b",
    "no_edema": r"\b(no pulmonary edema|without pulmonary edema|negative for pulmonary edema)\b",
}


CONCEPT_GROUPS = {
    "respiratory_abnormality": [
        "opacity",
        "consolidation",
        "infiltrate",
        "edema",
        "effusion",
        "pneumothorax",
        "atelectasis",
        "pneumonia",
        "bilateral",
        "low_volume",
    ],
    "support_device": [
        "tube",
        "ventilator",
        "cpap_bipap",
        "line_device",
    ],
    "criticality": [
        "resp_failure",
        "hypoxia",
        "shock",
        "sepsis",
        "arrest",
        "icu",
        "severe",
    ],
    "cardiac_hemodynamic": [
        "cardiomegaly",
        "vascular_congestion",
        "heart_failure",
        "tachycardia",
        "hypotension",
    ],
    "negative_clean": [
        "clear_lungs",
        "no_effusion",
        "no_pneumothorax",
        "no_edema",
    ],
}


TASK_CONCEPT_PRIORS = {
    "mortality_risk": {
        "criticality": 1.30,
        "support_device": 0.95,
        "cardiac_hemodynamic": 0.85,
        "respiratory_abnormality": 0.55,
        "negative_clean": -0.55,
    },
    "icu_risk": {
        "criticality": 0.90,
        "support_device": 0.75,
        "respiratory_abnormality": 0.55,
        "cardiac_hemodynamic": 0.50,
        "negative_clean": -0.40,
    },
    "ventilation_risk": {
        "support_device": 1.00,
        "criticality": 0.70,
        "respiratory_abnormality": 0.55,
        "cardiac_hemodynamic": 0.25,
        "negative_clean": -0.35,
    },
}


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def normalize_study_id(x) -> str:
    s = str(x).strip()
    if s.lower().endswith(".json"):
        s = s[:-5]
    return s


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_pred_item(data: dict, task: str) -> dict:
    preds = data.get("predictions", [])
    if isinstance(preds, list):
        for item in preds:
            if isinstance(item, dict) and item.get("task_name") == task:
                return item
    return {}


def parse_rationale(rationale: str) -> dict:
    out = {}
    for chunk in str(rationale or "").split(","):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        out[k.strip()] = safe_float(v.strip(), 0.0)
    return out


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    return r / float(len(x) - 1)


def soft_weighted_label(scores: np.ndarray, labels: np.ndarray, tau: float) -> float:
    if len(scores) == 0:
        return 0.0
    s = scores.astype(float) - float(np.max(scores))
    w = np.exp(s / max(tau, 1e-8))
    z = float(w.sum())
    if z <= 0:
        return float(labels.mean())
    return float((w / z * labels).sum())


def sharpen_prob(p: np.ndarray, gamma: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    a = np.power(p, gamma)
    b = np.power(1.0 - p, gamma)
    return a / (a + b)


def text_blob_from_report_json(data: dict, task: str) -> str:
    parts = []
    pred = get_pred_item(data, task)
    parts.append(str(pred.get("rationale", "")))

    retrieval = data.get("retrievals", {}).get(task, {})
    if isinstance(retrieval, dict):
        parts.append(str(retrieval.get("query_text", "")))
        neighbors = retrieval.get("neighbors", [])
        if isinstance(neighbors, list):
            for n in neighbors[:5]:
                if isinstance(n, dict):
                    parts.append(" ".join(str(x) for x in n.get("match_reasons", [])))

    conflict = data.get("conflicts", {}).get(task, {})
    if isinstance(conflict, dict):
        parts.append(" ".join(str(x) for x in conflict.get("conflict_sources", [])))

    return " ".join(parts).lower()


def extract_concept_features(text: str, task: str) -> dict:
    feats = {}
    hits = {}

    text = str(text or "").lower()

    for name, pat in CONCEPT_PATTERNS.items():
        found = 1.0 if re.search(pat, text) else 0.0
        hits[name] = found
        feats[f"concept_{name}"] = found

    for group, names in CONCEPT_GROUPS.items():
        vals = [hits.get(n, 0.0) for n in names]
        feats[f"concept_group_{group}_count"] = float(np.sum(vals))
        feats[f"concept_group_{group}_any"] = 1.0 if np.sum(vals) > 0 else 0.0

    prior = 0.0
    priors = TASK_CONCEPT_PRIORS.get(task, {})
    for group, weight in priors.items():
        prior += weight * feats.get(f"concept_group_{group}_count", 0.0)

    feats["concept_task_prior_raw"] = float(prior)
    feats["concept_task_prior_tanh"] = float(np.tanh(prior / 3.0))

    feats["concept_support_x_critical"] = (
        feats.get("concept_group_support_device_any", 0.0)
        * feats.get("concept_group_criticality_any", 0.0)
    )
    feats["concept_resp_x_critical"] = (
        feats.get("concept_group_respiratory_abnormality_any", 0.0)
        * feats.get("concept_group_criticality_any", 0.0)
    )
    feats["concept_negative_x_no_support"] = (
        feats.get("concept_group_negative_clean_any", 0.0)
        * (1.0 - feats.get("concept_group_support_device_any", 0.0))
    )

    return feats


def summarize_neighbors(labels: List[float], scores: List[float]) -> dict:
    if not labels:
        out = {}
        for k in [1, 2, 3, 5, 10, 12]:
            out[f"top{k}_label_mean"] = 0.0
            out[f"top{k}_pos_count"] = 0.0
            out[f"top{k}_score_mean"] = 0.0
        out.update(
            {
                "top1_label": 0.0,
                "top1_score": 0.0,
                "label_mean": 0.0,
                "label_std": 0.0,
                "label_min": 0.0,
                "label_max": 0.0,
                "score_mean": 0.0,
                "score_std": 0.0,
                "score_max": 0.0,
                "score_min": 0.0,
                "score_range": 0.0,
                "score_gap_1_2": 0.0,
                "score_gap_1_3": 0.0,
                "score_gap_1_5": 0.0,
                "score_gap_1_mean": 0.0,
                "weighted_label_tau_001": 0.0,
                "weighted_label_tau_003": 0.0,
                "weighted_label_tau_005": 0.0,
                "weighted_label_tau_010": 0.0,
                "weighted_label_tau_020": 0.0,
                "weighted_label_tau_050": 0.0,
                "label_disagreement": 0.0,
                "top5_vs_all_label_gap": 0.0,
                "top1_vs_top5_label_gap": 0.0,
            }
        )
        return out

    labels_arr = np.asarray(labels, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)

    out = {
        "top1_label": float(labels_arr[0]),
        "top1_score": float(scores_arr[0]),
        "label_mean": float(labels_arr.mean()),
        "label_std": float(labels_arr.std()),
        "label_min": float(labels_arr.min()),
        "label_max": float(labels_arr.max()),
        "score_mean": float(scores_arr.mean()),
        "score_std": float(scores_arr.std()),
        "score_max": float(scores_arr.max()),
        "score_min": float(scores_arr.min()),
        "score_range": float(scores_arr.max() - scores_arr.min()),
    }

    for k in [1, 2, 3, 5, 10, 12]:
        kk = min(k, len(labels_arr))
        out[f"top{k}_label_mean"] = float(labels_arr[:kk].mean())
        out[f"top{k}_pos_count"] = float(labels_arr[:kk].sum())
        out[f"top{k}_score_mean"] = float(scores_arr[:kk].mean())

    out["score_gap_1_2"] = float(scores_arr[0] - scores_arr[1]) if len(scores_arr) >= 2 else 0.0
    out["score_gap_1_3"] = float(scores_arr[0] - scores_arr[2]) if len(scores_arr) >= 3 else 0.0
    out["score_gap_1_5"] = float(scores_arr[0] - scores_arr[4]) if len(scores_arr) >= 5 else 0.0
    out["score_gap_1_mean"] = float(scores_arr[0] - scores_arr.mean())

    for tau in [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]:
        key = str(tau).replace(".", "")
        out[f"weighted_label_tau_{key}"] = soft_weighted_label(scores_arr, labels_arr, tau)

    p = float(labels_arr.mean())
    out["label_disagreement"] = float(p * (1.0 - p))
    out["top5_vs_all_label_gap"] = float(out["top5_label_mean"] - out["label_mean"])
    out["top1_vs_top5_label_gap"] = float(out["top1_label"] - out["top5_label_mean"])
    return out


def extract_task_features(data: dict, task: str) -> dict:
    pred = get_pred_item(data, task)

    point = safe_float(pred.get("point"), 0.0)
    interval_low = safe_float(pred.get("interval_low"), point)
    interval_high = safe_float(pred.get("interval_high"), point)
    interval_width = max(0.0, interval_high - interval_low)
    abstain = 1.0 if bool(pred.get("abstain", False)) else 0.0

    retrieval = data.get("retrievals", {}).get(task, {})
    if not isinstance(retrieval, dict):
        retrieval = {}

    neighbors = retrieval.get("neighbors", [])
    if not isinstance(neighbors, list):
        neighbors = []

    labels, scores = [], []
    reason_text = 0
    reason_image = 0
    reason_hybrid = 0

    for n in neighbors:
        if not isinstance(n, dict):
            continue
        labels.append(safe_float(n.get("label_value"), 0.0))
        scores.append(safe_float(n.get("score"), 0.0))
        reasons = n.get("match_reasons", [])
        if isinstance(reasons, list):
            rs = " ".join(str(x).lower() for x in reasons)
        else:
            rs = str(reasons).lower()
        reason_text += 1 if "text" in rs else 0
        reason_image += 1 if "image" in rs else 0
        reason_hybrid += 1 if "hybrid" in rs else 0

    neigh = summarize_neighbors(labels, scores)

    conflict = data.get("conflicts", {}).get(task, {})
    if not isinstance(conflict, dict):
        conflict = {}

    conflict_score = safe_float(conflict.get("conflict_score"), 0.0)
    conflict_sources = conflict.get("conflict_sources", [])
    if not isinstance(conflict_sources, list):
        conflict_sources = []

    conflict_source_text = " ".join(str(x).lower() for x in conflict_sources)
    conflict_vector = conflict.get("conflict_vector", [])
    if isinstance(conflict_vector, list) and conflict_vector:
        cv = np.asarray([safe_float(x) for x in conflict_vector], dtype=float)
        conflict_vec_mean = float(cv.mean())
        conflict_vec_std = float(cv.std())
        conflict_vec_max = float(cv.max())
        conflict_vec_l2 = float(np.linalg.norm(cv))
    else:
        conflict_vec_mean = 0.0
        conflict_vec_std = 0.0
        conflict_vec_max = 0.0
        conflict_vec_l2 = 0.0

    verification = data.get("verifications", {}).get(task, {})
    if not isinstance(verification, dict):
        verification = {}

    verification_confidence = safe_float(verification.get("confidence"), point)
    verification_passed = 1.0 if bool(verification.get("passed", True)) else 0.0

    rat = parse_rationale(pred.get("rationale", ""))

    feats = {
        "point": point,
        "point_sq": point * point,
        "point_sqrt": np.sqrt(max(point, 0.0)),
        "point_logit_like": np.log((point + 1e-5) / (1.0 - point + 1e-5)),
        "point_margin_05": abs(point - 0.5),
        "point_sharp_13": float(sharpen_prob(np.array([point]), 1.3)[0]),
        "interval_low": interval_low,
        "interval_high": interval_high,
        "interval_width": interval_width,
        "abstain": abstain,
        "retrieval_median": safe_float(retrieval.get("median"), 0.0),
        "retrieval_q10": safe_float(retrieval.get("q10"), 0.0),
        "retrieval_q90": safe_float(retrieval.get("q90"), 0.0),
        "retrieval_iqr": safe_float(retrieval.get("q90"), 0.0) - safe_float(retrieval.get("q10"), 0.0),
        "retrieval_std": safe_float(retrieval.get("std"), 0.0),
        "retrieval_effective_n": safe_float(retrieval.get("effective_n"), len(labels)),
        "retrieval_insufficient": 1.0 if bool(retrieval.get("insufficient_support", False)) else 0.0,
        "retrieval_unstable": 1.0 if bool(retrieval.get("unstable_distribution", False)) else 0.0,
        "reason_text_count": float(reason_text),
        "reason_image_count": float(reason_image),
        "reason_hybrid_count": float(reason_hybrid),
        "conflict_score": conflict_score,
        "conflict_num_sources": float(len(conflict_sources)),
        "conflict_has_gap": 1.0 if "gap" in conflict_source_text else 0.0,
        "conflict_has_mismatch": 1.0 if "mismatch" in conflict_source_text else 0.0,
        "conflict_has_unstable": 1.0 if "unstable" in conflict_source_text else 0.0,
        "conflict_has_insufficient": 1.0 if "insufficient" in conflict_source_text else 0.0,
        "conflict_vec_mean": conflict_vec_mean,
        "conflict_vec_std": conflict_vec_std,
        "conflict_vec_max": conflict_vec_max,
        "conflict_vec_l2": conflict_vec_l2,
        "verification_confidence": verification_confidence,
        "verification_passed": verification_passed,
        "rat_evidence": rat.get("evidence", 0.0),
        "rat_evidence_top": rat.get("evidence_top", 0.0),
        "rat_vision": rat.get("vision", 0.0),
        "rat_retrieval": rat.get("retrieval", 0.0),
        "rat_top1_label": rat.get("top1_label", 0.0),
        "rat_top1_score": rat.get("top1_score", 0.0),
        "rat_coverage": rat.get("coverage", 0.0),
        "rat_conflict": rat.get("conflict", conflict_score),
        "rat_support_n": rat.get("support_n", len(labels)),
    }

    feats.update(neigh)
    concept_text = text_blob_from_report_json(data, task)
    concept_feats = extract_concept_features(concept_text, task)
    feats.update(concept_feats)

    feats["point_x_retrieval"] = feats["point"] * feats["retrieval_median"]
    feats["point_x_top5"] = feats["point"] * feats["top5_label_mean"]
    feats["point_x_concept_prior"] = feats["point"] * feats["concept_task_prior_tanh"]
    feats["retrieval_x_conflict"] = feats["retrieval_median"] * feats["conflict_score"]
    feats["top5_x_scoregap"] = feats["top5_label_mean"] * feats["score_gap_1_mean"]
    feats["concept_prior_x_top5"] = feats["concept_task_prior_tanh"] * feats["top5_label_mean"]
    feats["concept_prior_x_retrieval"] = feats["concept_task_prior_tanh"] * feats["retrieval_median"]
    feats["confidence_gap"] = feats["verification_confidence"] - feats["point"]

    return feats


def extract_all_task_points(data: dict) -> Dict[str, float]:
    out = {}
    for task in TASKS:
        pred = get_pred_item(data, task)
        out[task] = safe_float(pred.get("point"), 0.0)
    return out


def build_table(reports_dir: Path, labels_csv: Path) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)
    labels_df["study_id"] = labels_df["study_id"].astype(str).map(normalize_study_id)

    rows = []
    for fp in sorted(reports_dir.glob("*.json")):
        try:
            data = load_json(fp)
        except Exception:
            continue

        sid = normalize_study_id(data.get("study_id", fp.stem))
        row = {"study_id": sid}
        task_points = extract_all_task_points(data)

        for task in TASKS:
            feats = extract_task_features(data, task)
            for k, v in feats.items():
                row[f"{task}__{k}"] = v

            other_tasks = [t for t in TASKS if t != task]
            other_mean = float(np.mean([task_points[t] for t in other_tasks]))

            row[f"{task}__other_point_mean"] = other_mean
            row[f"{task}__point_minus_other_mean"] = float(task_points[task] - other_mean)
            row[f"{task}__mortality_point"] = float(task_points["mortality_risk"])
            row[f"{task}__icu_point"] = float(task_points["icu_risk"])
            row[f"{task}__ventilation_point"] = float(task_points["ventilation_risk"])
            row[f"{task}__mortality_minus_icu"] = float(task_points["mortality_risk"] - task_points["icu_risk"])
            row[f"{task}__icu_minus_ventilation"] = float(task_points["icu_risk"] - task_points["ventilation_risk"])
            row[f"{task}__mortality_minus_ventilation"] = float(task_points["mortality_risk"] - task_points["ventilation_risk"])

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    return labels_df.merge(pred_df, on="study_id", how="inner")


def get_xy(df: pd.DataFrame, task: str):
    feat_cols = [c for c in df.columns if c.startswith(f"{task}__")]
    xdf = (
        df[feat_cols]
        .astype(float)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .copy()
    )

    for base_col in [
        f"{task}__point",
        f"{task}__retrieval_median",
        f"{task}__top5_label_mean",
        f"{task}__concept_task_prior_tanh",
        f"{task}__verification_confidence",
    ]:
        if base_col in xdf.columns:
            vals = xdf[base_col].to_numpy(dtype=float)
            xdf[f"{base_col}__rank01"] = rank01(vals)
            xdf[f"{base_col}__centered"] = vals - float(np.mean(vals))
            xdf[f"{base_col}__abs_centered"] = np.abs(xdf[f"{base_col}__centered"].to_numpy(dtype=float))

    X = xdf.to_numpy()
    y = df[task].astype(int).to_numpy()
    return X, y, list(xdf.columns)


def sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n = len(y)
    pos = max(int(y.sum()), 1)
    neg = max(n - pos, 1)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.2, class_weight="balanced", solver="lbfgs"),
        )
    if kind == "lr_wide":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=6000, C=2.5, class_weight="balanced", solver="lbfgs"),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=800, max_depth=10, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=1000, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=380, learning_rate=0.020, max_depth=2,
            subsample=0.90, random_state=42
        )
    if kind == "gb3":
        return GradientBoostingClassifier(
            n_estimators=220, learning_rate=0.025, max_depth=3,
            subsample=0.85, random_state=42
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=460, learning_rate=0.022, max_leaf_nodes=15,
            l2_regularization=0.01, random_state=42
        )
    if kind == "hgb_deep":
        return HistGradientBoostingClassifier(
            max_iter=640, learning_rate=0.018, max_leaf_nodes=31,
            l2_regularization=0.005, random_state=42
        )
    if kind == "stack_lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=6000, C=1.8, class_weight="balanced", solver="lbfgs"),
        )
    if kind == "stack_hgb":
        return HistGradientBoostingClassifier(
            max_iter=320, learning_rate=0.03, max_leaf_nodes=15,
            l2_regularization=0.01, random_state=42
        )
    if kind == "mortality_stack_hgb":
        return HistGradientBoostingClassifier(
            max_iter=420, learning_rate=0.022, max_leaf_nodes=31,
            l2_regularization=0.003, random_state=42
        )
    raise ValueError(f"Unknown model kind: {kind}")


def fit_predict_proba(model, kind: str, Xtr, ytr, Xva):
    if kind in {"gb", "gb3", "hgb", "hgb_deep", "stack_hgb", "mortality_stack_hgb"}:
        model.fit(Xtr, ytr, sample_weight=sample_weights(ytr))
    else:
        model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1]


def oof_model(df: pd.DataFrame, task: str, kind: str, n_splits: int = 5) -> np.ndarray:
    X, y, _ = get_xy(df, task)
    oof = np.zeros(len(df), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr, va in skf.split(X, y):
        model = make_model(kind)
        oof[va] = fit_predict_proba(model, kind, X[tr], y[tr], X[va])

    return oof


def build_meta_features(score_dict: Dict[str, np.ndarray], task: str) -> pd.DataFrame:
    base = {k: np.asarray(v, dtype=float) for k, v in score_dict.items() if k in BASE_MODELS}
    df = pd.DataFrame(base)

    arr = np.column_stack([base[k] for k in BASE_MODELS])
    df["mean_all"] = arr.mean(axis=1)
    df["std_all"] = arr.std(axis=1)
    df["max_all"] = arr.max(axis=1)
    df["min_all"] = arr.min(axis=1)
    df["range_all"] = df["max_all"] - df["min_all"]

    tree_cols = ["rf", "et", "gb", "gb3", "hgb", "hgb_deep"]
    tree_arr = np.column_stack([base[k] for k in tree_cols])
    df["mean_tree"] = tree_arr.mean(axis=1)
    df["std_tree"] = tree_arr.std(axis=1)
    df["max_tree"] = tree_arr.max(axis=1)
    df["min_tree"] = tree_arr.min(axis=1)
    df["range_tree"] = df["max_tree"] - df["min_tree"]

    for c in list(base.keys()) + ["mean_all", "mean_tree"]:
        vals = df[c].to_numpy(dtype=float)
        df[f"{c}_rank"] = rank01(vals)
        df[f"{c}_margin05"] = np.abs(vals - 0.5)

    df["gb_minus_lr"] = df["gb"] - df["lr"]
    df["gb3_minus_gb"] = df["gb3"] - df["gb"]
    df["hgb_minus_gb"] = df["hgb"] - df["gb"]
    df["hgbdeep_minus_hgb"] = df["hgb_deep"] - df["hgb"]
    df["et_minus_rf"] = df["et"] - df["rf"]
    df["tree_minus_lr"] = df["mean_tree"] - df["lr"]
    df["tree_rank_minus_lr_rank"] = df["mean_tree_rank"] - df["lr_rank"]

    for c in ["mean_all", "mean_tree", "gb3", "hgb", "hgb_deep"]:
        vals = df[c].to_numpy(dtype=float)
        df[f"{c}_sharp_13"] = sharpen_prob(vals, 1.3)
        df[f"{c}_sharp_18"] = sharpen_prob(vals, 1.8)

    if task == "mortality_risk":
        keep = [
            "gb", "gb3", "hgb", "hgb_deep", "et",
            "mean_all", "mean_tree", "std_all", "std_tree", "range_all", "range_tree",
            "gb_minus_lr", "gb3_minus_gb", "hgb_minus_gb", "hgbdeep_minus_hgb",
            "gb_rank", "gb3_rank", "hgb_rank", "hgb_deep_rank", "et_rank",
            "mean_tree_rank", "mean_all_rank",
            "gb3_sharp_13", "gb3_sharp_18", "hgb_sharp_13", "hgb_deep_sharp_13",
            "mean_tree_sharp_13", "mean_tree_sharp_18",
        ]
        keep = [c for c in keep if c in df.columns]
        return df[keep].astype(float)

    return df.astype(float)


def oof_stack_from_base(base_oof: Dict[str, np.ndarray], y: np.ndarray, kind: str, task: str, n_splits: int = 5) -> np.ndarray:
    X_meta = build_meta_features(base_oof, task).to_numpy()
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr, va in skf.split(X_meta, y):
        model = make_model(kind)
        oof[va] = fit_predict_proba(model, kind, X_meta[tr], y[tr], X_meta[va])

    return oof


def full_stack_from_base(base_full: Dict[str, np.ndarray], y: np.ndarray, kind: str, task: str) -> np.ndarray:
    X_meta = build_meta_features(base_full, task).to_numpy()
    model = make_model(kind)
    if kind in {"stack_hgb", "mortality_stack_hgb"}:
        model.fit(X_meta, y, sample_weight=sample_weights(y))
    else:
        model.fit(X_meta, y)
    return model.predict_proba(X_meta)[:, 1]


def metrics_at_threshold(y_true, y_score, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)

    out = {
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Acc": accuracy_score(y_true, y_pred),
    }
    if len(np.unique(y_true)) >= 2:
        out["AUROC"] = roc_auc_score(y_true, y_score)
        out["AUPRC"] = average_precision_score(y_true, y_score)
    else:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan
    return out


def scan_best_f1(y_true, y_score) -> dict:
    best = {"thr": 0.5, "F1": -1.0, "Acc": -1.0}
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics_at_threshold(y_true, y_score, thr)
        if (m["F1"] > best["F1"]) or (abs(m["F1"] - best["F1"]) < 1e-12 and m["Acc"] > best["Acc"]):
            best = {"thr": float(thr), "F1": float(m["F1"]), "Acc": float(m["Acc"])}
    return best


def scan_best_acc(y_true, y_score) -> dict:
    best = {"thr": 0.5, "F1": -1.0, "Acc": -1.0}
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics_at_threshold(y_true, y_score, thr)
        if (m["Acc"] > best["Acc"]) or (abs(m["Acc"] - best["Acc"]) < 1e-12 and m["F1"] > best["F1"]):
            best = {"thr": float(thr), "F1": float(m["F1"]), "Acc": float(m["Acc"])}
    return best


def evaluate_score(task: str, y: np.ndarray, score: np.ndarray, model_name: str) -> dict:
    m05 = metrics_at_threshold(y, score, 0.5)
    best_f1 = scan_best_f1(y, score)
    best_acc = scan_best_acc(y, score)
    return {
        "task": task,
        "n": len(y),
        "AUROC": m05["AUROC"],
        "AUPRC": m05["AUPRC"],
        "F1_0.5": m05["F1"],
        "Acc_0.5": m05["Acc"],
        "best_f1_threshold": best_f1["thr"],
        "best_F1": best_f1["F1"],
        "Acc_at_best_F1": best_f1["Acc"],
        "best_acc_threshold": best_acc["thr"],
        "F1_at_best_Acc": best_acc["F1"],
        "best_Acc": best_acc["Acc"],
        "positive_rate": float(np.mean(y)),
        "model": model_name,
        "eval_mode": "oof_cv",
    }


def add_task_best_rows(df: pd.DataFrame, metric: str, model_name: str) -> pd.DataFrame:
    rows = []
    base = df[df["task"].isin(TASKS)].copy()
    for task in TASKS:
        g = base[base["task"] == task].sort_values(metric, ascending=False)
        if not g.empty:
            r = g.iloc[0].copy()
            r["model"] = model_name
            rows.append(r)
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def add_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for model_name, g in df[df["task"].isin(TASKS)].groupby("model"):
        summary.append(
            {
                "task": "MEAN",
                "n": int(g["n"].mean()),
                "AUROC": float(g["AUROC"].mean()),
                "AUPRC": float(g["AUPRC"].mean()),
                "F1_0.5": float(g["F1_0.5"].mean()),
                "Acc_0.5": float(g["Acc_0.5"].mean()),
                "best_f1_threshold": np.nan,
                "best_F1": float(g["best_F1"].mean()),
                "Acc_at_best_F1": float(g["Acc_at_best_F1"].mean()),
                "best_acc_threshold": np.nan,
                "F1_at_best_Acc": float(g["F1_at_best_Acc"].mean()),
                "best_Acc": float(g["best_Acc"].mean()),
                "positive_rate": float(g["positive_rate"].mean()),
                "model": model_name,
                "eval_mode": "oof_cv_mean",
            }
        )
    return pd.concat([df, pd.DataFrame(summary)], ignore_index=True)


def task_specific_blend(task: str, score_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(score_dict)

    if task == "mortality_risk":
        out["task_blend_auc"] = (
            0.28 * score_dict["mortality_stack_hgb"]
            + 0.18 * score_dict["stack_hgb"]
            + 0.18 * score_dict["gb3"]
            + 0.14 * score_dict["hgb_deep"]
            + 0.12 * score_dict["ens_concept_heavy"]
            + 0.10 * score_dict["ens_auc"]
        )
        out["task_blend_acc"] = (
            0.22 * score_dict["stack_lr"]
            + 0.18 * score_dict["ens_f1_stable"]
            + 0.18 * score_dict["ens_acc_stable"]
            + 0.16 * score_dict["gb3"]
            + 0.14 * score_dict["hgb_deep"]
            + 0.12 * score_dict["mortality_stack_hgb"]
        )
        out["task_blend_joint"] = 0.62 * out["task_blend_auc"] + 0.38 * out["task_blend_acc"]
    elif task == "icu_risk":
        out["task_blend_auc"] = (
            0.30 * score_dict["stack_hgb"]
            + 0.24 * score_dict["ens_concept_heavy"]
            + 0.18 * score_dict["ens_auc"]
            + 0.14 * score_dict["stack_lr"]
            + 0.14 * score_dict["rank_ens_auc"]
        )
        out["task_blend_acc"] = (
            0.26 * score_dict["stack_lr"]
            + 0.20 * score_dict["ens_acc_stable"]
            + 0.18 * score_dict["gb3"]
            + 0.18 * score_dict["hgb_deep"]
            + 0.18 * score_dict["ens_f1_stable"]
        )
        out["task_blend_joint"] = 0.50 * out["task_blend_auc"] + 0.50 * out["task_blend_acc"]
    else:
        out["task_blend_auc"] = (
            0.26 * score_dict["stack_hgb"]
            + 0.22 * score_dict["rank_ens_auc"]
            + 0.18 * score_dict["ens_auc"]
            + 0.18 * score_dict["ens_concept_heavy"]
            + 0.16 * score_dict["stack_lr"]
        )
        out["task_blend_acc"] = (
            0.26 * score_dict["stack_lr"]
            + 0.22 * score_dict["ens_acc_stable"]
            + 0.18 * score_dict["gb3"]
            + 0.18 * score_dict["ens_f1_stable"]
            + 0.16 * score_dict["hgb_deep"]
        )
        out["task_blend_joint"] = 0.50 * out["task_blend_auc"] + 0.50 * out["task_blend_acc"]

    return out


def run_oof_eval(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    all_rows = []

    for task in TASKS:
        print(f"[TASK] {task}")
        _, y, _ = get_xy(df, task)

        oof = {}
        for kind in BASE_MODELS + ["lr_wide"]:
            print(f"  [OOF] {kind}")
            oof[kind] = oof_model(df, task, kind, n_splits=n_splits)

        oof["ens_tree_balanced"] = (
            0.16 * oof["rf"] + 0.22 * oof["et"] + 0.18 * oof["gb"] + 0.20 * oof["gb3"] + 0.24 * oof["hgb"]
        )
        oof["ens_auc"] = (
            0.05 * oof["lr"] + 0.08 * oof["rf"] + 0.17 * oof["et"] + 0.20 * oof["gb"] +
            0.20 * oof["gb3"] + 0.20 * oof["hgb"] + 0.10 * oof["hgb_deep"]
        )
        oof["ens_f1_stable"] = (
            0.08 * oof["lr"] + 0.12 * oof["lr_wide"] + 0.16 * oof["rf"] + 0.20 * oof["et"] +
            0.14 * oof["gb"] + 0.16 * oof["gb3"] + 0.14 * oof["hgb"]
        )
        oof["ens_acc_stable"] = (
            0.08 * oof["lr"] + 0.14 * oof["lr_wide"] + 0.14 * oof["rf"] + 0.18 * oof["et"] +
            0.14 * oof["gb"] + 0.16 * oof["gb3"] + 0.16 * oof["hgb"]
        )
        oof["ens_concept_heavy"] = (
            0.08 * oof["lr"] + 0.10 * oof["lr_wide"] + 0.12 * oof["rf"] + 0.18 * oof["et"] +
            0.16 * oof["gb"] + 0.18 * oof["gb3"] + 0.18 * oof["hgb"]
        )

        rank = {k: rank01(v) for k, v in oof.items() if k in BASE_MODELS + ["lr_wide"]}
        oof["rank_ens_auc"] = (
            0.05 * rank["lr"] + 0.08 * rank["rf"] + 0.17 * rank["et"] + 0.20 * rank["gb"] +
            0.20 * rank["gb3"] + 0.20 * rank["hgb"] + 0.10 * rank["hgb_deep"]
        )
        oof["rank_ens_balanced"] = (
            0.08 * rank["lr"] + 0.12 * rank["lr_wide"] + 0.14 * rank["rf"] + 0.18 * rank["et"] +
            0.15 * rank["gb"] + 0.16 * rank["gb3"] + 0.17 * rank["hgb"]
        )

        base_for_stack = {k: oof[k] for k in BASE_MODELS}
        oof["stack_lr"] = oof_stack_from_base(base_for_stack, y, "stack_lr", task, n_splits=n_splits)
        oof["stack_hgb"] = oof_stack_from_base(base_for_stack, y, "stack_hgb", task, n_splits=n_splits)

        if task == "mortality_risk":
            oof["mortality_stack_hgb"] = oof_stack_from_base(base_for_stack, y, "mortality_stack_hgb", task, n_splits=n_splits)

        oof = task_specific_blend(task, oof)

        for name, score in oof.items():
            all_rows.append(evaluate_score(task, y, score, name))

    out = pd.DataFrame(all_rows)
    out = add_task_best_rows(out, "AUROC", "TASK_BEST_AUROC")
    out = add_task_best_rows(out, "AUPRC", "TASK_BEST_AUPRC")
    out = add_task_best_rows(out, "Acc_0.5", "TASK_BEST_ACC_0.5")
    out = add_task_best_rows(out, "F1_0.5", "TASK_BEST_F1_0.5")
    out = add_task_best_rows(out, "best_Acc", "TASK_BEST_BEST_ACC")
    out = add_summary_rows(out)
    return out


def fit_full_model(df: pd.DataFrame, task: str, kind: str) -> np.ndarray:
    X, y, _ = get_xy(df, task)
    model = make_model(kind)
    if kind in {"gb", "gb3", "hgb", "hgb_deep", "stack_hgb", "mortality_stack_hgb"}:
        model.fit(X, y, sample_weight=sample_weights(y))
    else:
        model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def full_scores_for_task(df: pd.DataFrame, task: str) -> Dict[str, np.ndarray]:
    _, y, _ = get_xy(df, task)
    scores = {}

    for kind in BASE_MODELS + ["lr_wide"]:
        scores[kind] = fit_full_model(df, task, kind)

    scores["ens_tree_balanced"] = (
        0.16 * scores["rf"] + 0.22 * scores["et"] + 0.18 * scores["gb"] + 0.20 * scores["gb3"] + 0.24 * scores["hgb"]
    )
    scores["ens_auc"] = (
        0.05 * scores["lr"] + 0.08 * scores["rf"] + 0.17 * scores["et"] + 0.20 * scores["gb"] +
        0.20 * scores["gb3"] + 0.20 * scores["hgb"] + 0.10 * scores["hgb_deep"]
    )
    scores["ens_f1_stable"] = (
        0.08 * scores["lr"] + 0.12 * scores["lr_wide"] + 0.16 * scores["rf"] + 0.20 * scores["et"] +
        0.14 * scores["gb"] + 0.16 * scores["gb3"] + 0.14 * scores["hgb"]
    )
    scores["ens_acc_stable"] = (
        0.08 * scores["lr"] + 0.14 * scores["lr_wide"] + 0.14 * scores["rf"] + 0.18 * scores["et"] +
        0.14 * scores["gb"] + 0.16 * scores["gb3"] + 0.16 * scores["hgb"]
    )
    scores["ens_concept_heavy"] = (
        0.08 * scores["lr"] + 0.10 * scores["lr_wide"] + 0.12 * scores["rf"] + 0.18 * scores["et"] +
        0.16 * scores["gb"] + 0.18 * scores["gb3"] + 0.18 * scores["hgb"]
    )

    rank = {k: rank01(v) for k, v in scores.items() if k in BASE_MODELS + ["lr_wide"]}
    scores["rank_ens_auc"] = (
        0.05 * rank["lr"] + 0.08 * rank["rf"] + 0.17 * rank["et"] + 0.20 * rank["gb"] +
        0.20 * rank["gb3"] + 0.20 * rank["hgb"] + 0.10 * rank["hgb_deep"]
    )
    scores["rank_ens_balanced"] = (
        0.08 * rank["lr"] + 0.12 * rank["lr_wide"] + 0.14 * rank["rf"] + 0.18 * rank["et"] +
        0.15 * rank["gb"] + 0.16 * rank["gb3"] + 0.17 * rank["hgb"]
    )

    base_for_stack = {k: scores[k] for k in BASE_MODELS}
    scores["stack_lr"] = full_stack_from_base(base_for_stack, y, "stack_lr", task)
    scores["stack_hgb"] = full_stack_from_base(base_for_stack, y, "stack_hgb", task)

    if task == "mortality_risk":
        scores["mortality_stack_hgb"] = full_stack_from_base(base_for_stack, y, "mortality_stack_hgb", task)

    scores = task_specific_blend(task, scores)
    return scores


def select_best_models(metrics: pd.DataFrame, criterion: str) -> Dict[str, str]:
    if criterion == "task_best_auroc":
        metric = "AUROC"
    elif criterion == "task_best_auprc":
        metric = "AUPRC"
    elif criterion == "task_best_acc":
        metric = "Acc_0.5"
    elif criterion == "task_best_f1":
        metric = "F1_0.5"
    elif criterion == "task_best_best_acc":
        metric = "best_Acc"
    else:
        return {task: criterion for task in TASKS}

    selected = {}
    base = metrics[metrics["task"].isin(TASKS)].copy()
    base = base[~base["model"].str.startswith("TASK_BEST")]
    for task in TASKS:
        g = base[base["task"] == task].sort_values(metric, ascending=False)
        selected[task] = str(g.iloc[0]["model"])
    return selected


def fit_all_and_write(
    df: pd.DataFrame,
    reports_dir: Path,
    out_dir: Path,
    write_model: str,
    metrics: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_best_models(metrics, write_model)
    print("write model selection:", selected)

    all_scores: Dict[str, Dict[str, float]] = {}

    for task in TASKS:
        print(f"[FIT-FULL] {task}")
        score_dict = full_scores_for_task(df, task)
        model_name = selected[task]
        if model_name not in score_dict:
            raise RuntimeError(f"Unknown write model {model_name}; available={sorted(score_dict)}")

        pred = score_dict[model_name]
        for sid, score in zip(df["study_id"].astype(str).tolist(), pred.tolist()):
            all_scores.setdefault(sid, {})[task] = float(score)

    for fp in sorted(reports_dir.glob("*.json")):
        data = load_json(fp)
        sid = normalize_study_id(data.get("study_id", fp.stem))
        if sid not in all_scores:
            continue

        preds = data.get("predictions", [])
        if isinstance(preds, list):
            for item in preds:
                if not isinstance(item, dict):
                    continue
                task = item.get("task_name")
                if task not in TASKS:
                    continue

                old = safe_float(item.get("point"), 0.0)
                new = all_scores[sid][task]
                item["raw_point_before_calibration"] = old
                item["point"] = float(new)
                item["interval_low"] = max(0.0, float(new) - 0.12)
                item["interval_high"] = min(1.0, float(new) + 0.12)
                item["rationale"] = str(item.get("rationale", "")) + f", concept_structured_calibrated_{selected[task]}={new:.3f}"

        save_json(data, out_dir / fp.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--save-csv", default="artifacts/concept_structured_calibrator_metrics.csv")
    parser.add_argument("--out-dir", default="artifacts/reports_calibrated_concept_structured")
    parser.add_argument("--write-calibrated", action="store_true")
    parser.add_argument(
        "--write-model",
        default="task_best_auroc",
        choices=[
            "lr", "lr_wide", "rf", "et", "gb", "gb3", "hgb", "hgb_deep",
            "ens_tree_balanced", "ens_auc", "ens_f1_stable", "ens_acc_stable",
            "ens_concept_heavy", "rank_ens_auc", "rank_ens_balanced",
            "stack_lr", "stack_hgb", "mortality_stack_hgb",
            "task_blend_auc", "task_blend_acc", "task_blend_joint",
            "task_best_auroc", "task_best_auprc", "task_best_acc", "task_best_f1", "task_best_best_acc",
        ],
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    df = build_table(Path(args.reports_dir), Path(args.labels_csv))
    print("matched samples:", len(df))

    metrics = run_oof_eval(df, n_splits=args.folds)
    metrics.to_csv(args.save_csv, index=False)

    print(metrics.to_string(index=False))
    print(f"Saved metrics to {args.save_csv}")

    if args.write_calibrated:
        fit_all_and_write(
            df=df,
            reports_dir=Path(args.reports_dir),
            out_dir=Path(args.out_dir),
            write_model=args.write_model,
            metrics=metrics,
        )
        print(f"Saved calibrated reports to {args.out_dir}")


if __name__ == "__main__":
    main()