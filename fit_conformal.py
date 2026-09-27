from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from mm_rcna.config import load_config
from mm_rcna.calibrate.conformal import SplitConformalCalibrator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--pred-csv', required=True)
    p.add_argument('--output-json', default='./artifacts/conformal.json')
    args = p.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(args.pred_csv)
    cal = SplitConformalCalibrator(alpha=cfg.conformal.alpha, qhat=cfg.conformal.qhat)
    cal.fit(df['y_true'].to_numpy(), df['y_pred'].to_numpy())
    out = {'alpha': float(cfg.conformal.alpha), 'qhat': float(cal.qhat), 'n_calib': int(len(df)), 'source_csv': args.pred_csv}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
