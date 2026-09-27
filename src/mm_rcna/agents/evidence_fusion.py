from __future__ import annotations

from typing import List, Tuple

from mm_rcna.config import AppConfig
from mm_rcna.schemas import EvidenceItem


class EvidenceFusion:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

    def run(self, evidence: List[EvidenceItem]) -> Tuple[List[EvidenceItem], float]:
        if not evidence:
            return [], 0.0
        merged = {}
        for item in evidence:
            key = (item.topic, item.finding)
            if key not in merged or item.score > merged[key].score:
                merged[key] = item
        fused = list(merged.values())
        coverage = min(1.0, len(fused) / 6.0)
        return fused, float(coverage)
