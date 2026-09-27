from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--labels-csv', required=True, help='CSV with lesion and region weak labels aggregated by study or image')
    p.add_argument('--out', default='./artifacts/vision_heads.json')
    args = p.parse_args()
    df = pd.read_csv(args.labels_csv)
    lesion_cols = [c for c in df.columns if c.startswith('lesion_')]
    region_cols = [c for c in df.columns if c.startswith('region_')]
    payload = {
        'lesion_bias': {c.replace('lesion_', ''): float(df[c].mean()) for c in lesion_cols},
        'region_bias': {c.replace('region_', ''): float(df[c].mean()) for c in region_cols},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
