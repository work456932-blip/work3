
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from mm_rcna.config import AppConfig
from mm_rcna.schemas import ImageRef, StudyRecord
from mm_rcna.data.report_reader import read_text_if_exists


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}


def _clean_id(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


class MIMICCXRStudyBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

        p = Path(config.data.studies_csv)
        if p.exists():
            try:
                self.df = pd.read_csv(p)
            except Exception:
                self.df = pd.DataFrame(columns=["study_id", "subject_id"])
        else:
            self.df = pd.DataFrame(columns=["study_id", "subject_id"])

    def _get_subject_id_from_df(self, study_id: str) -> Optional[str]:
        if self.df.empty or "study_id" not in self.df.columns:
            return None
        row = self.df[self.df["study_id"].astype(str) == str(study_id)]
        if row.empty:
            return None
        row0 = row.iloc[0]
        if "subject_id" not in row0:
            return None
        return _clean_id(row0["subject_id"])

    def _infer_subject_id_from_raw_tree(self, image_root: Path, study_id: str) -> Optional[str]:
        study_dir_name = f"s{study_id}"
        for study_dir in image_root.glob(f"p*/p*/{study_dir_name}"):
            if not study_dir.is_dir():
                continue
            patient_dir = study_dir.parent.name
            if patient_dir.startswith("p") and len(patient_dir) > 1:
                return patient_dir[1:]
        return None

    def _list_flat_images(self, image_root: Path, study_id: str) -> List[Path]:
        out: List[Path] = []
        for candidate in sorted(image_root.glob(f"{study_id}_*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                out.append(candidate)
        return out

    def _list_raw_images(self, image_root: Path, study_id: str, subject_id: Optional[str]) -> List[Path]:
        out: List[Path] = []
        study_id = _clean_id(study_id)
        subject_id = _clean_id(subject_id)

        if subject_id and subject_id.isdigit() and len(subject_id) >= 2:
            exact_dir = image_root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}"
            if exact_dir.exists() and exact_dir.is_dir():
                for candidate in sorted(exact_dir.iterdir()):
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                        out.append(candidate)
                if out:
                    return out

        for candidate in sorted(image_root.glob(f"p*/p*/s{study_id}/*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                out.append(candidate)
        return out

    def _read_flat_txt(self, root: Path, study_id: str) -> str:
        return read_text_if_exists(root / f"{study_id}.txt")

    def _read_raw_txt(self, root: Path, study_id: str, subject_id: Optional[str]) -> str:
        study_id = _clean_id(study_id)
        subject_id = _clean_id(subject_id)

        if subject_id and subject_id.isdigit() and len(subject_id) >= 2:
            pfx = f"p{subject_id[:2]}"
            pdir = f"p{subject_id}"

            direct_candidates = [
                root / pfx / pdir / f"s{study_id}.txt",
                root / pfx / pdir / f"{study_id}.txt",
            ]
            for p in direct_candidates:
                txt = read_text_if_exists(p)
                if txt:
                    return txt

            nested_dir = root / pfx / pdir / f"s{study_id}"
            if nested_dir.exists() and nested_dir.is_dir():
                for p in sorted(nested_dir.glob("*.txt")):
                    txt = read_text_if_exists(p)
                    if txt:
                        return txt

        for p in sorted(root.glob(f"p*/p*/s{study_id}.txt")):
            txt = read_text_if_exists(p)
            if txt:
                return txt
        for p in sorted(root.glob(f"p*/p*/{study_id}.txt")):
            txt = read_text_if_exists(p)
            if txt:
                return txt
        for p in sorted(root.glob(f"p*/p*/s{study_id}/*.txt")):
            txt = read_text_if_exists(p)
            if txt:
                return txt
        return ""

    def _resolve_raw_root(self, preferred: str, fallback_prepared: str) -> tuple[Path | None, Path]:
        raw_root = Path(preferred) if preferred else None
        prepared_root = Path(fallback_prepared)
        return raw_root, prepared_root

    def _load_notes(self, study_id: str, subject_id: Optional[str]) -> str:
        raw_root, prepared_root = self._resolve_raw_root(
            self.cfg.data.raw_notes_dir, self.cfg.data.notes_dir
        )
        if self.cfg.data.prefer_raw_data and raw_root is not None:
            txt = self._read_raw_txt(raw_root, study_id, subject_id)
            if txt:
                return txt
        txt = self._read_flat_txt(prepared_root, study_id)
        if txt:
            return txt
        if (not self.cfg.data.prefer_raw_data) and raw_root is not None:
            return self._read_raw_txt(raw_root, study_id, subject_id)
        return ""

    def _load_report(self, study_id: str, subject_id: Optional[str]) -> str:
        raw_root, prepared_root = self._resolve_raw_root(
            self.cfg.data.raw_reports_dir, self.cfg.data.reports_dir
        )
        if self.cfg.data.prefer_raw_data and raw_root is not None:
            txt = self._read_raw_txt(raw_root, study_id, subject_id)
            if txt:
                return txt
        txt = self._read_flat_txt(prepared_root, study_id)
        if txt:
            return txt
        if (not self.cfg.data.prefer_raw_data) and raw_root is not None:
            return self._read_raw_txt(raw_root, study_id, subject_id)
        return ""

    def build_study(self, study_id: str) -> StudyRecord:
        study_id = _clean_id(study_id)

        raw_image_root = Path(self.cfg.data.raw_images_dir) if self.cfg.data.raw_images_dir else None
        prepared_image_root = Path(self.cfg.data.images_dir)

        subject_id = self._get_subject_id_from_df(study_id)

        if not subject_id and raw_image_root is not None:
            subject_id = self._infer_subject_id_from_raw_tree(raw_image_root, study_id)

        raw_images: List[Path] = []
        if self.cfg.data.prefer_raw_data and raw_image_root is not None:
            raw_images = self._list_raw_images(raw_image_root, study_id, subject_id)

        flat_images: List[Path] = []
        if not raw_images:
            flat_images = self._list_flat_images(prepared_image_root, study_id)

        if (not self.cfg.data.prefer_raw_data) and not flat_images and raw_image_root is not None:
            raw_images = self._list_raw_images(raw_image_root, study_id, subject_id)

        final_images = raw_images if raw_images else flat_images

        notes = self._load_notes(study_id, subject_id)
        report = self._load_report(study_id, subject_id)
        images = [ImageRef(path=str(p), view="unknown") for p in final_images]

        return StudyRecord(
            study_id=str(study_id),
            subject_id=subject_id,
            notes=notes,
            report=report,
            images=images,
        )
