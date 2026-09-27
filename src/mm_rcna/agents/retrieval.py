from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mm_rcna.config import AppConfig
from mm_rcna.schemas import ConflictReport, EvidenceItem, RetrievalNeighbor, RetrievalSummary
from mm_rcna.models.text_encoder import TextEncoder
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.utils.io_utils import load_pickle


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def _softmax_weights(scores: np.ndarray, tau: float = 0.03) -> np.ndarray:
    if scores.size == 0:
        return scores
    s = scores.astype(np.float32)
    s = s - s.max()
    w = np.exp(s / max(tau, 1e-6))
    z = w.sum()
    if z <= 0:
        return np.ones_like(w) / max(len(w), 1)
    return w / z


class RetrievalAgent:
    def __init__(self, config: AppConfig, device: str = "cpu") -> None:
        self.cfg = config
        idx_dir = Path(config.data.index_dir)

        self.meta = load_pickle(idx_dir / "meta.pkl", default=[])
        self.text_vecs = (
            _normalize_rows(np.load(idx_dir / "text.npy"))
            if (idx_dir / "text.npy").exists()
            else np.zeros((0, 64), dtype=np.float32)
        )
        self.image_vecs = (
            _normalize_rows(np.load(idx_dir / "image.npy"))
            if (idx_dir / "image.npy").exists()
            else np.zeros((0, 64), dtype=np.float32)
        )

        self.text_encoder = TextEncoder(config.models.text_encoder_name, device=device)
        self.vision = VisionToolRunner(
            config.models.image_encoder_name,
            config.models.vision_backbone_name,
            config.models.lesion_labels,
            config.models.lung_regions,
            image_size=config.models.image_size,
            device=device,
            checkpoint_path=config.models.vision_checkpoint if config.models.use_trained_vision_heads else None,
        )

    def _query_text(self, items: List[EvidenceItem], task_name: str) -> str:
        rel = [x for x in items if task_name in x.supports_tasks]
        if not rel:
            return f"{task_name}: empty evidence"

        rel = sorted(rel, key=lambda x: float(x.score), reverse=True)[:12]
        chunks = [f"{x.topic}: {x.finding} ({x.modality}, {float(x.score):.2f})" for x in rel]
        return " ; ".join(chunks)

    def retrieve(
        self,
        task_name: str,
        label_column: str,
        evidence: List[EvidenceItem],
        image_paths: List[str],
        query_study_id: Optional[str] = None,
        query_subject_id: Optional[str] = None,
        override_top_k: Optional[int] = None,
    ) -> RetrievalSummary:
        text_query = self._query_text(evidence, task_name)
        txt = _normalize_vec(self.text_encoder.encode([text_query])[0])

        vision_out = self.vision.run(image_paths)
        img = _normalize_vec(np.asarray(vision_out.feature_vector, dtype=np.float32))

        k = int(override_top_k or self.cfg.retrieval.top_k)
        score_map: Dict[int, float] = {}
        reason_map: Dict[int, List[str]] = {}

        if len(self.text_vecs):
            txt_scores = self.text_vecs @ txt
            for idx, score in enumerate(txt_scores.tolist()):
                wscore = self.cfg.retrieval.text_weight * float(score)
                score_map[idx] = score_map.get(idx, 0.0) + wscore
                if abs(wscore) > 1e-8:
                    reason_map.setdefault(idx, []).append("text")

        if len(self.image_vecs):
            img_scores = self.image_vecs @ img
            for idx, score in enumerate(img_scores.tolist()):
                wscore = self.cfg.retrieval.image_weight * float(score)
                score_map[idx] = score_map.get(idx, 0.0) + wscore
                if abs(wscore) > 1e-8:
                    reason_map.setdefault(idx, []).append("image")

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        neighbors: List[RetrievalNeighbor] = []
        label_values: List[float] = []
        hybrid_scores: List[float] = []

        for idx, score in ranked:
            m = self.meta[idx]

            if query_study_id is not None and str(m.get("study_id")) == str(query_study_id):
                continue
            if (
                self.cfg.retrieval.exclude_same_subject
                and query_subject_id is not None
                and str(m.get("subject_id")) == str(query_subject_id)
            ):
                continue

            lv = float(m.get("labels", {}).get(label_column, m.get("labels", {}).get(task_name, 0.0)))
            label_values.append(lv)
            hybrid_scores.append(float(score))

            reasons = reason_map.get(idx, ["hybrid"])
            neighbors.append(
                RetrievalNeighbor(
                    study_id=str(m.get("study_id")),
                    score=float(score),
                    match_reasons=reasons,
                    label_value=lv,
                )
            )
            if len(neighbors) >= k:
                break

        if label_values:
            labels_arr = np.asarray(label_values, dtype=np.float32)
            score_arr = np.asarray(hybrid_scores, dtype=np.float32)
            weights = _softmax_weights(score_arr, tau=0.03)

            weighted_pos = float(np.sum(weights * labels_arr))
            weighted_var = float(np.sum(weights * (labels_arr - weighted_pos) ** 2))
            weighted_std = float(np.sqrt(max(weighted_var, 0.0)))

            q10 = float(np.quantile(labels_arr, 0.10))
            q90 = float(np.quantile(labels_arr, 0.90))

            # 更保守的“不稳定”定义：标签混合 + 离散高
            unstable = (weighted_std > self.cfg.contracts.unstable_dist_std_threshold) and (0.15 < weighted_pos < 0.85)
        else:
            labels_arr = np.asarray([0.0], dtype=np.float32)
            score_arr = np.asarray([0.0], dtype=np.float32)
            weighted_pos = 0.0
            weighted_std = 0.0
            q10 = 0.0
            q90 = 0.0
            unstable = False

        return RetrievalSummary(
            task_name=task_name,
            neighbors=neighbors,
            median=weighted_pos,
            q10=q10,
            q90=q90,
            std=weighted_std,
            effective_n=int(len(label_values)),
            insufficient_support=len(label_values) < self.cfg.retrieval.min_effective_n,
            unstable_distribution=unstable,
            query_text=text_query,
        )


class ConflictRetrievalAgent:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        idx_dir = Path(config.data.index_dir)
        self.meta = load_pickle(idx_dir / "meta.pkl", default=[])

    def retrieve(
        self,
        task_name: str,
        retrieval_summary: RetrievalSummary,
        conflict: ConflictReport,
        label_column: Optional[str] = None,
        query_study_id: Optional[str] = None,
        query_subject_id: Optional[str] = None,
    ) -> RetrievalSummary:
        vals = []
        neighbors = []

        for m in self.meta:
            if query_study_id is not None and str(m.get("study_id")) == str(query_study_id):
                continue
            if (
                self.cfg.retrieval.exclude_same_subject
                and query_subject_id is not None
                and str(m.get("subject_id")) == str(query_subject_id)
            ):
                continue

            label_value = float(m.get("labels", {}).get(label_column or task_name, retrieval_summary.median))
            mismatch = abs(label_value - retrieval_summary.median)
            score = max(0.0, 1.0 - abs(mismatch - conflict.conflict_score))

            vals.append(label_value)
            neighbors.append(
                RetrievalNeighbor(
                    study_id=str(m.get("study_id")),
                    score=float(score),
                    match_reasons=conflict.conflict_sources,
                    label_value=label_value,
                )
            )

        neighbors = sorted(neighbors, key=lambda x: x.score, reverse=True)[: self.cfg.retrieval.conflict_top_k]
        vals = [x.label_value for x in neighbors]
        scores = [x.score for x in neighbors]

        if vals:
            arr = np.asarray(vals, dtype=np.float32)
            sarr = np.asarray(scores, dtype=np.float32)
            weights = _softmax_weights(sarr, tau=0.05)
            weighted_pos = float(np.sum(weights * arr))
            weighted_var = float(np.sum(weights * (arr - weighted_pos) ** 2))
            weighted_std = float(np.sqrt(max(weighted_var, 0.0)))
        else:
            arr = np.asarray([retrieval_summary.median], dtype=np.float32)
            weighted_pos = float(retrieval_summary.median)
            weighted_std = 0.0

        return RetrievalSummary(
            task_name=task_name,
            neighbors=neighbors,
            median=weighted_pos,
            q10=float(np.quantile(arr, 0.1)),
            q90=float(np.quantile(arr, 0.9)),
            std=weighted_std,
            effective_n=len(vals),
            insufficient_support=len(vals) < max(3, self.cfg.retrieval.min_effective_n // 2),
            unstable_distribution=(weighted_std > self.cfg.contracts.unstable_dist_std_threshold) and (0.15 < weighted_pos < 0.85),
            query_text="conflict-pattern",
        )