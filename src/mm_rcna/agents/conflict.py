from __future__ import annotations

import numpy as np

from mm_rcna.schemas import ConflictReport, RetrievalSummary, VisionOutput


class MediatorAgent:
    def __init__(self, llm_client=None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model

    def run(self, task_name: str, fused_evidence, retrieval: RetrievalSummary, vision_out: VisionOutput | None):
        task_scores = [float(x.score) for x in (fused_evidence or []) if task_name in getattr(x, "supports_tasks", [])]
        all_scores = [float(x.score) for x in (fused_evidence or [])]

        evidence_score = float(np.mean(task_scores)) if task_scores else (float(np.mean(all_scores)) if all_scores else 0.0)
        retrieval_uncertainty = float(retrieval.std)

        quality_penalty = 0.0
        if vision_out is not None:
            quality_penalty = min(0.30, 0.06 * len(vision_out.quality_flags))

        text_scores = [float(x.score) for x in (fused_evidence or []) if getattr(x, "modality", "") == "text"]
        vis_scores = [float(x.score) for x in (fused_evidence or []) if getattr(x, "modality", "") == "vision"]

        cross_modal_gap = 0.0
        if text_scores and vis_scores:
            cross_modal_gap = abs(float(np.mean(text_scores)) - float(np.mean(vis_scores)))

        evidence_retrieval_gap = abs(evidence_score - float(retrieval.median))
        insufficient_penalty = 0.12 if retrieval.insufficient_support else 0.0
        unstable_penalty = 0.18 if retrieval.unstable_distribution else 0.0

        conflict_score = float(np.clip(
            0.42 * evidence_retrieval_gap
            + 0.24 * retrieval_uncertainty
            + 0.18 * cross_modal_gap
            + quality_penalty
            + insufficient_penalty
            + unstable_penalty,
            0.0,
            1.0,
        ))

        sources = []
        if evidence_retrieval_gap > 0.18:
            sources.append("evidence_retrieval_gap")
        if cross_modal_gap > 0.18:
            sources.append("vision_text_mismatch")
        if retrieval.unstable_distribution:
            sources.append("retrieval_distribution_unstable")
        if retrieval.insufficient_support:
            sources.append("insufficient_support")
        if quality_penalty > 0:
            sources.append("vision_quality_issue")
        if not sources:
            sources.append("low_conflict")

        conflict_vector = np.asarray([
            evidence_score,
            float(retrieval.median),
            float(retrieval.q10),
            float(retrieval.q90),
            float(retrieval.std),
            float(retrieval.effective_n),
            evidence_retrieval_gap,
            cross_modal_gap,
            quality_penalty,
            1.0 if retrieval.insufficient_support else 0.0,
            1.0 if retrieval.unstable_distribution else 0.0,
            conflict_score,
        ], dtype=np.float32)

        if conflict_vector.shape[0] < 64:
            conflict_vector = np.concatenate(
                [conflict_vector, np.zeros(64 - conflict_vector.shape[0], dtype=np.float32)],
                axis=0,
            )
        else:
            conflict_vector = conflict_vector[:64]

        norm = np.linalg.norm(conflict_vector)
        if norm > 0:
            conflict_vector = conflict_vector / norm

        return ConflictReport(
            task_name=task_name,
            conflict_score=conflict_score,
            conflict_sources=sources,
            conflict_vector=conflict_vector.tolist(),
            trigger_conflict_retrieval=conflict_score >= 0.65,
        )
