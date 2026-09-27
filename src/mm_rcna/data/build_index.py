from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mm_rcna.config import load_config
from mm_rcna.models.text_encoder import TextEncoder
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.utils.io_utils import save_pickle, ensure_dir


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _load_studies_with_fallback(cfg) -> pd.DataFrame:
    studies = _safe_read_csv(cfg.data.studies_csv)

    if not studies.empty and "study_id" in studies.columns:
        return studies

    labels = _safe_read_csv(cfg.data.labels_csv)
    if labels.empty or "study_id" not in labels.columns:
        raise RuntimeError(
            f"Both studies_csv and labels_csv are unusable.\n"
            f"studies_csv={cfg.data.studies_csv}\n"
            f"labels_csv={cfg.data.labels_csv}"
        )

    keep_cols = [c for c in ["study_id", "subject_id", "notes", "report"] if c in labels.columns]
    studies = labels[keep_cols].copy()
    studies = studies.drop_duplicates(subset=["study_id"]).reset_index(drop=True)

    Path(cfg.data.studies_csv).parent.mkdir(parents=True, exist_ok=True)
    studies.to_csv(cfg.data.studies_csv, index=False)

    print(f"[INFO] studies.csv rebuilt from labels.csv -> {cfg.data.studies_csv}")
    print(f"[INFO] rebuilt studies shape = {studies.shape}")

    return studies


def _build_label_map(cfg) -> dict:
    labels = _safe_read_csv(cfg.data.labels_csv)
    if labels.empty or "study_id" not in labels.columns:
        return {}

    labels = labels.drop_duplicates(subset=["study_id"]).reset_index(drop=True)
    return labels.set_index("study_id").to_dict(orient="index")


def build_index(config_path: str, device: str = "cpu") -> None:
    cfg = load_config(config_path)
    ensure_dir(cfg.data.index_dir)

    studies = _load_studies_with_fallback(cfg)
    label_map = _build_label_map(cfg)

    if studies.empty:
        raise RuntimeError("No studies available after fallback. Cannot build index.")

    text_encoder = TextEncoder(cfg.models.text_encoder_name, device=device)
    vision = VisionToolRunner(
        cfg.models.image_encoder_name,
        cfg.models.vision_backbone_name,
        cfg.models.lesion_labels,
        cfg.models.lung_regions,
        cfg.models.image_size,
        device=device,
        checkpoint_path=cfg.models.vision_checkpoint if cfg.models.use_trained_vision_heads else None,
    )

    metas = []
    text_vecs = []
    image_vecs = []

    notes_dir = Path(cfg.data.notes_dir)
    reports_dir = Path(cfg.data.reports_dir)
    images_dir = Path(cfg.data.images_dir)

    for _, row in tqdm(studies.iterrows(), total=len(studies), desc="Building index"):
        sid = str(row["study_id"])
        subject_id = None if "subject_id" not in row or pd.isna(row["subject_id"]) else str(row["subject_id"])

        note_path = notes_dir / f"{sid}.txt"
        report_path = reports_dir / f"{sid}.txt"

        if note_path.exists():
            note_text = note_path.read_text(encoding="utf-8", errors="ignore")
        else:
            note_text = "" if "notes" not in row or pd.isna(row["notes"]) else str(row["notes"])

        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        else:
            report_text = "" if "report" not in row or pd.isna(row["report"]) else str(row["report"])

        text = (note_text + " " + report_text).strip()
        tvec = text_encoder.encode([text])[0].astype(np.float32)

        image_paths = sorted([str(p) for p in images_dir.glob(f"{sid}_*")])
        vout = vision.run(image_paths)
        ivec = np.asarray(vout.feature_vector, dtype=np.float32)

        text_vecs.append(tvec)
        image_vecs.append(ivec)

        metas.append(
            {
                "study_id": sid,
                "subject_id": subject_id,
                "labels": label_map.get(sid, {}),
                "image_paths": image_paths,
                "quality_flags": list(vout.quality_flags),
                "lesion_scores": dict(vout.lesion_scores),
                "region_scores": dict(vout.region_scores),
                "text_encoder_name": cfg.models.text_encoder_name,
                "image_encoder_name": cfg.models.image_encoder_name,
                "vision_backbone_name": cfg.models.vision_backbone_name,
            }
        )

    text_arr = np.asarray(text_vecs, dtype=np.float32)
    image_arr = np.asarray(image_vecs, dtype=np.float32)

    np.save(Path(cfg.data.index_dir) / "text.npy", text_arr)
    np.save(Path(cfg.data.index_dir) / "image.npy", image_arr)
    save_pickle(metas, Path(cfg.data.index_dir) / "meta.pkl")

    print(f"Index saved to {cfg.data.index_dir}")
    print(f"Built {len(metas)} studies")
    print(f"text.npy shape = {text_arr.shape}")
    print(f"image.npy shape = {image_arr.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    build_index(args.config, device=args.device)


if __name__ == "__main__":
    main()