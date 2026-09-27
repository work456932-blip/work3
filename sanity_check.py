from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from mm_rcna.config import load_config
from mm_rcna.utils.io_utils import ensure_dir


def _bootstrap_demo_files(cfg):
    ensure_dir(cfg.paths.artifacts_dir)
    ensure_dir(cfg.data.notes_dir)
    ensure_dir(cfg.data.reports_dir)
    ensure_dir(cfg.data.images_dir)
    ensure_dir(cfg.data.index_dir)
    studies_csv = Path(cfg.data.studies_csv)
    labels_csv = Path(cfg.data.labels_csv)
    if not studies_csv.exists():
        pd.DataFrame([{'study_id': 'demo-study-001', 'subject_id': 'demo-subject-001'}]).to_csv(studies_csv, index=False)
    if not labels_csv.exists():
        pd.DataFrame([{'study_id': 'demo-study-001', 'mortality_risk': 0.3, 'icu_risk': 0.4, 'ventilation_risk': 0.35}]).to_csv(labels_csv, index=False)
    note = Path(cfg.data.notes_dir) / 'demo-study-001.txt'
    if not note.exists():
        note.write_text('Patient is unstable and may need ICU support. Possible respiratory failure.', encoding='utf-8')
    report = Path(cfg.data.reports_dir) / 'demo-study-001.txt'
    if not report.exists():
        report.write_text('Portable chest radiograph shows bibasal opacity and small pleural effusion.', encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    _bootstrap_demo_files(cfg)
    print('Sanity-check assets prepared.')
    print('Next steps:')
    print(f'  python -m src.mm_rcna.data.build_index --config {args.config}')
    print(f'  python run_pipeline.py --config {args.config} --study-id demo-study-001')


if __name__ == '__main__':
    main()
