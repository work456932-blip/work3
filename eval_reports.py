from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)

TASKS = ["mortality_risk", "icu_risk", "ventilation_risk"]

def load_report_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_prediction_map(report_data: dict) -> dict[str, float]:
    out = {}
    preds = report_data.get("predictions", [])
    if isinstance(preds, list):
        for item in preds:
            if isinstance(item, dict) and item.get("task_name") in TASKS:
                task = item.get("task_name")
                point = item.get("point")
                if task and point is not None:
                    out[task] = float(point)
    return out

def scan_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    best_acc = -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and acc > best_acc):
            best_f1 = float(f1)
            best_acc = float(acc)
            best_thr = float(thr)
    return best_thr, best_f1, best_acc

def evaluate_reports(reports_dir: str, labels_csv: str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv)
    labels_df["study_id"] = labels_df["study_id"].apply(str)

    rows = []
    for fp in sorted(Path(reports_dir).glob("*.json")):
        try:
            data = load_report_json(fp)
        except Exception:
            continue
        study_id = str(data.get("study_id", fp.stem))
        pred_map = extract_prediction_map(data)
        if not pred_map:
            continue
        row = {"study_id": study_id}
        row.update(pred_map)
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    merged = labels_df.merge(pred_df, on="study_id", how="inner", suffixes=("_true", "_pred"))

    metrics = []
    for task in TASKS:
        y_true = merged[f"{task}_true"].values
        y_score = merged[f"{task}_pred"].values

        # 固定阈值 0.5
        y_pred_05 = (y_score >= 0.5).astype(int)
        f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
        acc_05 = float(accuracy_score(y_true, y_pred_05))

        # 最优阈值
        best_thr, best_f1, best_acc = scan_best_threshold(y_true, y_score)

        metrics.append(
            {
                "task": task,
                "n": len(y_true),
                "F1_0.5": f1_05,
                "Acc_0.5": acc_05,
                "best_threshold": best_thr,
                "best_F1": best_f1,
                "best_Acc_at_best_F1": best_acc,
            }
        )

    return pd.DataFrame(metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--save-csv", required=True)
    args = parser.parse_args()

    df = evaluate_reports(args.reports_dir, args.labels_csv)
    df.to_csv(args.save_csv, index=False)
    print(f"Saved to {args.save_csv}")

if __name__ == "__main__":
    main()