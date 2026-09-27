from __future__ import annotations

import numpy as np


class SplitConformalCalibrator:
    def __init__(self, alpha: float = 0.1, qhat: float = 0.15) -> None:
        self.alpha = float(alpha)
        self.qhat = float(qhat)

    def fit(self, y_true, y_pred) -> None:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        scores = np.abs(yt - yp)
        if len(scores) == 0:
            self.qhat = float(self.qhat)
            return
        self.qhat = float(np.quantile(scores, 1 - self.alpha, method='higher'))

    def predict_interval(self, point: float) -> tuple[float, float]:
        lo = max(0.0, float(point) - self.qhat)
        hi = min(1.0, float(point) + self.qhat)
        return lo, hi
