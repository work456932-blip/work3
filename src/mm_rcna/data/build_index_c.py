from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mm_rcna.config import load_config
from mm_rcna.models.text_encoder import TextEncoder
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.utils.io_utils import save_pickle, ensure_dir


def build_index(config_path: str) -> None:
    print(f"[INFO] Loading config from: {config_path}")
    cfg = load_config(config_path)

    print(f"[INFO] Index output dir: {cfg.data.index_dir}")
    ensure_dir(cfg.data.index_dir)

    print(f"[INFO] Reading studies CSV: {cfg.data.studies_csv}")
    studies = pd.read_csv(cfg.data.studies_csv)

    print(f"[INFO] Reading labels CSV: {cfg.data.labels_csv}")
    labels = (
        pd.read_csv(cfg.data.labels_csv)
        if Path(cfg.data.labels_csv).exists()
        else pd.DataFrame(columns=["study_id"])
    )
    if labels.empty:
        print("[WARN] labels.csv not found or empty, labels will be stored as empty dicts.")

    label_map = labels.set_index("study_id").to_dict(orient="index") if not labels.empty else {}

    print("[INFO] Initializing text encoder...")
    text_encoder = TextEncoder(cfg.models.text_encoder_name)

    print("[INFO] Initializing vision runner...")
    vision = VisionToolRunner(
        cfg.models.image_encoder_name,
        cfg.models.vision_backbone_name,
        cfg.models.lesion_labels,
        cfg.models.lung_regions,
        cfg.models.image_size,
        device="cpu",
    )

    total = len(studies)
    print(f"[INFO] Total studies to process: {total}")

    metas = []
    text_vecs = []
    image_vecs = []

    for idx, row in studies.iterrows():
        sid = str(row["study_id"])
        subject_id = None if "subject_id" not in row else str(row["subject_id"])

        print(f"\n[INFO] Processing {idx + 1}/{total}: study_id={sid}, subject_id={subject_id}")

        note_path = Path(cfg.data.notes_dir) / f"{sid}.txt"
        report_path = Path(cfg.data.reports_dir) / f"{sid}.txt"

        text = ""
        note_exists = note_path.exists()
        report_exists = report_path.exists()

        if note_exists:
            note_text = note_path.read_text(encoding="utf-8", errors="ignore")
            text += note_text + " "
            print(f"[INFO]   Note found: {note_path} (chars={len(note_text)})")
        else:
            print(f"[WARN]   Note not found: {note_path}")

        if report_exists:
            report_text = report_path.read_text(encoding="utf-8", errors="ignore")
            text += report_text
            print(f"[INFO]   Report found: {report_path} (chars={len(report_text)})")
        else:
            print(f"[WARN]   Report not found: {report_path}")

        print("[INFO]   Encoding text...")
        tvec = text_encoder.encode([text])[0]
        text_vecs.append(tvec)
        print(f"[INFO]   Text vector shape: {np.asarray(tvec).shape}")

        image_paths = [str(p) for p in Path(cfg.data.images_dir).glob(f"{sid}_*")]
        print(f"[INFO]   Found {len(image_paths)} image(s)")

        if len(image_paths) > 0:
            for p in image_paths:
                print(f"[INFO]     Image: {p}")
        else:
            print(f"[WARN]   No images found for study_id={sid}")

        print("[INFO]   Encoding images...")
        vout = vision.run(image_paths)
        ivec = np.asarray(vout.feature_vector, dtype=float)
        image_vecs.append(ivec)
        print(f"[INFO]   Image vector shape: {ivec.shape}")

        metas.append(
            {
                "study_id": sid,
                "subject_id": subject_id,
                "labels": label_map.get(sid, {}),
            }
        )

        print(f"[INFO]   Done: study_id={sid}")

    text_arr = np.asarray(text_vecs, dtype=np.float32)
    image_arr = np.asarray(image_vecs, dtype=np.float32)

    text_out = Path(cfg.data.index_dir) / "text.npy"
    image_out = Path(cfg.data.index_dir) / "image.npy"
    meta_out = Path(cfg.data.index_dir) / "meta.pkl"

    print("\n[INFO] Saving outputs...")
    print(f"[INFO]   text.npy shape: {text_arr.shape}")
    print(f"[INFO]   image.npy shape: {image_arr.shape}")
    print(f"[INFO]   meta count: {len(metas)}")

    np.save(text_out, text_arr)
    np.save(image_out, image_arr)
    save_pickle(metas, meta_out)

    print("[INFO] Build index finished successfully.")
    print(f"[INFO]   Saved text vectors to: {text_out}")
    print(f"[INFO]   Saved image vectors to: {image_out}")
    print(f"[INFO]   Saved metadata to: {meta_out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    build_index(args.config)


if __name__ == "__main__":
    main()