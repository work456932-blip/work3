from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--studies-csv', required=True)
    p.add_argument('--out-dir', default='./artifacts/batch_outputs')
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.studies_csv)
    for sid in df['study_id'].astype(str).tolist():
        cmd = [sys.executable, 'run_pipeline.py', '--config', args.config, '--study-id', sid, '--output-json', str(out_dir / f'{sid}.json')]
        subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
