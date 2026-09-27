from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mm_rcna.config import AppConfig


KEYWORDS = {
    "mortality_risk": [
        "expired",
        "death",
        "deceased",
        "code blue",
        "cardiac arrest",
        "critical",
    ],
    "icu_risk": [
        "icu",
        "intensive care",
        "vasopressor",
        "unstable",
        "shock",
        "critically ill",
    ],
    "ventilation_risk": [
        "intubated",
        "intubation",
        "ventilator",
        "mechanical ventilation",
        "respiratory failure",
        "ett",
        "endotracheal tube",
    ],
}


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    return ""


def _clean_id(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _score_from_text(text: str, task: str) -> float:
    low = (text or "").lower()
    hits = sum(1 for kw in KEYWORDS[task] if kw in low)
    if hits <= 0:
        return 0.0
    if hits == 1:
        return 0.5
    if hits == 2:
        return 0.75
    return 1.0


@dataclass
class AutoLabelBuilder:
    cfg: AppConfig

    def _load_studies_df(self) -> pd.DataFrame:
        p = Path(self.cfg.data.studies_csv)
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "study_id" in df.columns:
                    if "subject_id" not in df.columns:
                        df["subject_id"] = None
                    df["study_id"] = df["study_id"].map(_clean_id)
                    df["subject_id"] = df["subject_id"].map(_clean_id)
                    df = df[df["study_id"].notna()].drop_duplicates(subset=["study_id"])
                    return df[["study_id", "subject_id"]].reset_index(drop=True)
            except Exception:
                pass
        return pd.DataFrame(columns=["study_id", "subject_id"])

    def _infer_subject_id_from_report_path(self, report_path: Path) -> Optional[str]:
        # 支持 .../p10/p10000032/s50414267.txt 或类似层级
        parts = list(report_path.parts)
        for part in parts:
            if part.startswith("p") and part[1:].isdigit() and len(part) > 2:
                return part[1:]
        return None

    def _scan_report_files(self) -> List[Tuple[str, Optional[str], Path]]:
        roots = []
        if self.cfg.data.raw_reports_dir:
            roots.append(Path(self.cfg.data.raw_reports_dir))
        if self.cfg.data.raw_root:
            roots.append(Path(self.cfg.data.raw_root))

        seen = set()
        out: List[Tuple[str, Optional[str], Path]] = []

        for root in roots:
            if not root.exists():
                continue

            for p in root.rglob("*.txt"):
                name = p.stem
                study_id = None

                if name.startswith("s") and name[1:].isdigit():
                    study_id = name[1:]
                elif name.isdigit():
                    study_id = name
                else:
                    m = re.search(r"s(\d+)", str(p))
                    if m:
                        study_id = m.group(1)

                if not study_id:
                    continue

                key = (study_id, str(p))
                if key in seen:
                    continue
                seen.add(key)

                subject_id = self._infer_subject_id_from_report_path(p)
                out.append((study_id, subject_id, p))

        return out

    def _find_report_path(self, study_id: str, subject_id: Optional[str]) -> Optional[Path]:
        study_id = _clean_id(study_id)
        subject_id = _clean_id(subject_id)

        candidates: List[Path] = []
        roots = []
        if self.cfg.data.raw_reports_dir:
            roots.append(Path(self.cfg.data.raw_reports_dir))
        if self.cfg.data.raw_root:
            roots.append(Path(self.cfg.data.raw_root))

        for root in roots:
            if not root.exists():
                continue

            if subject_id and subject_id.isdigit() and len(subject_id) >= 2:
                pfx = f"p{subject_id[:2]}"
                pdir = f"p{subject_id}"
                candidates.extend(
                    [
                        root / pfx / pdir / f"s{study_id}.txt",
                        root / pfx / pdir / f"{study_id}.txt",
                        root / pfx / pdir / f"s{study_id}" / f"s{study_id}.txt",
                    ]
                )

            candidates.extend(
                [
                    root / f"s{study_id}.txt",
                    root / f"{study_id}.txt",
                ]
            )

            for c in candidates:
                if c.exists() and c.is_file():
                    return c

            glob_hits = list(root.glob(f"p*/p*/s{study_id}.txt"))
            if glob_hits:
                return glob_hits[0]

            glob_hits = list(root.glob(f"p*/p*/{study_id}.txt"))
            if glob_hits:
                return glob_hits[0]

            glob_hits = list(root.glob(f"p*/p*/s{study_id}/*.txt"))
            if glob_hits:
                return glob_hits[0]

        return None

    def _find_notes_text(self, study_id: str, subject_id: Optional[str]) -> str:
        study_id = _clean_id(study_id)
        subject_id = _clean_id(subject_id)

        roots = []
        if self.cfg.data.raw_notes_dir:
            roots.append(Path(self.cfg.data.raw_notes_dir))
        if self.cfg.data.raw_root:
            roots.append(Path(self.cfg.data.raw_root))

        for root in roots:
            if not root.exists():
                continue

            candidates = []
            if subject_id and subject_id.isdigit() and len(subject_id) >= 2:
                pfx = f"p{subject_id[:2]}"
                pdir = f"p{subject_id}"
                candidates.extend(
                    [
                        root / pfx / pdir / f"s{study_id}.txt",
                        root / pfx / pdir / f"{study_id}.txt",
                        root / pfx / pdir / f"s{study_id}" / f"s{study_id}.txt",
                    ]
                )

            candidates.extend([root / f"s{study_id}.txt", root / f"{study_id}.txt"])

            for c in candidates:
                txt = _read_text_if_exists(c)
                if txt:
                    return txt

        return ""

    def _build_source_table(self) -> pd.DataFrame:
        studies_df = self._load_studies_df()

        records = []

        if not studies_df.empty:
            for _, row in studies_df.iterrows():
                study_id = _clean_id(row.get("study_id"))
                subject_id = _clean_id(row.get("subject_id"))
                if not study_id:
                    continue

                report_path = self._find_report_path(study_id, subject_id)
                report = _read_text_if_exists(report_path) if report_path else ""
                notes = self._find_notes_text(study_id, subject_id)

                if not report and not notes:
                    continue

                records.append(
                    {
                        "study_id": study_id,
                        "subject_id": subject_id,
                        "notes": notes,
                        "report": report,
                    }
                )
        else:
            scanned = self._scan_report_files()
            for study_id, subject_id, report_path in scanned:
                report = _read_text_if_exists(report_path)
                notes = self._find_notes_text(study_id, subject_id)
                if not report and not notes:
                    continue

                records.append(
                    {
                        "study_id": _clean_id(study_id),
                        "subject_id": _clean_id(subject_id),
                        "notes": notes,
                        "report": report,
                    }
                )

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "study_id",
                    "subject_id",
                    "notes",
                    "report",
                    "mortality_risk",
                    "icu_risk",
                    "ventilation_risk",
                ]
            )

        df = df.drop_duplicates(subset=["study_id"]).reset_index(drop=True)
        merged = df["notes"].fillna("") + "\n" + df["report"].fillna("")

        for task in KEYWORDS:
            df[task] = merged.map(lambda x: _score_from_text(x, task))

        return df

    def build(self, output_csv: str) -> str:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self._build_source_table()
        df.to_csv(output_path, index=False)

        # 顺手补一份 studies.csv，方便后面 run_pipeline 直接查 study_id -> subject_id
        studies_out = Path(self.cfg.data.studies_csv)
        studies_out.parent.mkdir(parents=True, exist_ok=True)
        df[["study_id", "subject_id"]].drop_duplicates(subset=["study_id"]).to_csv(
            studies_out, index=False
        )

        return str(output_path)