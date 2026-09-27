from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from mm_rcna.config import load_config
from mm_rcna.data.mimic_cxr_dataset import MIMICCXRStudyBuilder


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else "", encoding="utf-8")


def to_plain_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported object type for serialization: {type(obj)}")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def package_one_study(study_id: str, output_dir: str, copy_files: bool = True) -> None:
    cfg = load_config("configs/default.yaml")
    builder = MIMICCXRStudyBuilder(cfg)

    study = builder.build_study(study_id)
    study_dict = to_plain_dict(study)

    out_root = Path(output_dir) / f"study_{study.study_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    json_path = out_root / "study_package.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(study_dict, f, indent=2, ensure_ascii=False)

    safe_write_text(out_root / "notes.txt", study.notes or "")
    safe_write_text(out_root / "report.txt", study.report or "")

    copied_images = []
    missing_images = []

    if copy_files:
        images_dir = out_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(study.images, start=1):
            src = Path(img.path)
            suffix = src.suffix if src.suffix else ".img"
            dst = images_dir / f"{study.study_id}_{i}{suffix}"
            ok = copy_if_exists(src, dst)
            if ok:
                copied_images.append(str(dst))
            else:
                missing_images.append(str(src))

    summary = {
        "study_id": study.study_id,
        "subject_id": study.subject_id,
        "notes_length": len(study.notes or ""),
        "report_length": len(study.report or ""),
        "num_images_found": len(study.images or []),
        "copied_images": copied_images,
        "missing_images": missing_images,
        "package_dir": str(out_root.resolve()),
        "json_manifest": str(json_path.resolve()),
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Study packaging completed")
    print("=" * 60)
    print(f"study_id         : {study.study_id}")
    print(f"subject_id       : {study.subject_id}")
    print(f"notes length     : {len(study.notes or '')}")
    print(f"report length    : {len(study.report or '')}")
    print(f"images found     : {len(study.images or [])}")
    print(f"package dir      : {out_root.resolve()}")
    print(f"manifest json    : {json_path.resolve()}")

    if copied_images:
        print("\nCopied images:")
        for p in copied_images:
            print(f"  - {p}")

    if missing_images:
        print("\nMissing images:")
        for p in missing_images:
            print(f"  - {p}")


def main():
    parser = argparse.ArgumentParser(description="Find and package one MIMIC-CXR study sample.")
    parser.add_argument("--study_id", required=True, help="Study ID to package")
    parser.add_argument(
        "--output_dir",
        default="artifacts/packaged_samples",
        help="Output directory for packaged sample",
    )
    parser.add_argument(
        "--no_copy_files",
        action="store_true",
        help="Do not copy image files, only generate manifest and text files",
    )

    args = parser.parse_args()

    package_one_study(
        study_id=args.study_id,
        output_dir=args.output_dir,
        copy_files=not args.no_copy_files,
    )


if __name__ == "__main__":
    main()