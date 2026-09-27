from __future__ import annotations

import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression


class LearnableContractModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit_from_csv(self, csv_path: str, out_path: str) -> None:
        df = pd.read_csv(csv_path)
        X = df[['conflict_score', 'retrieval_std', 'effective_n', 'interval_width']].to_numpy()
        y = df['passed'].astype(int).to_numpy()
        self.model.fit(X, y)
        with open(out_path, 'wb') as f:
            pickle.dump(self.model, f)
