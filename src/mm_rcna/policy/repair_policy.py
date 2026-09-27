from __future__ import annotations

from pathlib import Path
import pickle
from typing import List, Tuple


class RuleRepairPolicy:
    def choose_action(self, violations: List[str]) -> Tuple[str, str]:
        v = set(violations)
        if 'high_conflict' in v:
            return 'trigger_conflict_retrieval', 'Conflict too high'
        if 'insufficient_support' in v:
            return 'expand_standard_retrieval', 'Support is too weak'
        if 'interval_too_narrow' in v:
            return 'widen_interval', 'Need wider uncertainty interval'
        if 'low_coverage' in v:
            return 'rebuild_evidence', 'Coverage too low'
        return 'abstain', 'No safe repair action found'


class LearnedRepairPolicy:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.ready = self.path.exists()
        self.model = None
        if self.ready:
            try:
                with open(self.path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception:
                self.ready = False

    def choose_action(self, features: list[float]):
        if not self.ready or self.model is None:
            return 'abstain', 'Learned policy unavailable'
        pred = self.model.predict([features])[0]
        return str(pred), 'Learned repair policy'
