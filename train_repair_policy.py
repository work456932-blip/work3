from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--trace-csv', required=True)
    p.add_argument('--out', default='./artifacts/repair_policy.pkl')
    args = p.parse_args()
    df = pd.read_csv(args.trace_csv)
    X = df[['conflict_score', 'retrieval_std', 'effective_n', 'interval_width']].to_numpy()
    y = df['action'].astype(str).to_numpy()
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(clf, f)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
