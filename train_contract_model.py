from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from mm_rcna.agents.learnable_contract import LearnableContractModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--trace-csv', required=True)
    p.add_argument('--out', default='./artifacts/contract_model.pkl')
    args = p.parse_args()
    model = LearnableContractModel()
    model.fit_from_csv(args.trace_csv, args.out)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
