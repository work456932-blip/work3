from __future__ import annotations

from typing import Dict, List

from mm_rcna.schemas import ConflictReport, EvidenceItem, RetrievalSummary, TaskPrediction


class ExplainerAgent:
    def __init__(self, llm_client=None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model

    def _fallback_explain(
        self,
        study_id: str,
        predictions: List[TaskPrediction],
        evidence: List[EvidenceItem],
        retrievals: Dict[str, RetrievalSummary],
        conflicts: Dict[str, ConflictReport],
        audit_trace,
    ) -> str:
        lines = [f"Study {study_id}", "", "Key evidence:"]
        for e in evidence[:10]:
            lines.append(f"- [{e.topic}] {e.finding} (src={e.source}, score={e.score:.2f}, mod={e.modality})")
        lines.append("")
        for p in predictions:
            r = retrievals[p.task_name]
            c = conflicts[p.task_name]
            lines.append(
                f"{p.task_name}: point={p.point:.3f}, interval=[{p.interval_low:.3f}, {p.interval_high:.3f}], abstain={p.abstain}"
            )
            lines.append(
                f" support_n={r.effective_n}, retr_median={r.median:.3f}, retr_std={r.std:.3f}, conflict={c.conflict_score:.3f}"
            )
            lines.append("")
        lines.append(f"Audit steps: {len(audit_trace)}")
        return "\n".join(lines)

    def explain(
        self,
        study_id: str,
        predictions: List[TaskPrediction],
        evidence: List[EvidenceItem],
        retrievals: Dict[str, RetrievalSummary],
        conflicts: Dict[str, ConflictReport],
        audit_trace,
    ):
        fallback = self._fallback_explain(study_id, predictions, evidence, retrievals, conflicts, audit_trace)

        if self.llm_client is None or not getattr(self.llm_client, "ready", False) or not self.model:
            return fallback

        evidence_block = "\n".join(
            [f"- topic={e.topic}; finding={e.finding}; source={e.source}; score={e.score:.2f}; modality={e.modality}" for e in evidence[:12]]
        )

        pred_block = []
        for p in predictions:
            r = retrievals[p.task_name]
            c = conflicts[p.task_name]
            pred_block.append(
                f"{p.task_name}: point={p.point:.3f}, interval=[{p.interval_low:.3f},{p.interval_high:.3f}], "
                f"abstain={p.abstain}, support_n={r.effective_n}, retr_median={r.median:.3f}, "
                f"retr_std={r.std:.3f}, conflict={c.conflict_score:.3f}"
            )
        pred_block = "\n".join(pred_block)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful clinical AI reporting assistant. "
                    "Summarize conservatively. Do not invent findings. Mention uncertainty and conflicts explicitly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Study ID: {study_id}\n\n"
                    f"Evidence:\n{evidence_block}\n\n"
                    f"Predictions:\n{pred_block}\n\n"
                    f"Audit steps: {len(audit_trace)}\n\n"
                    f"Write a concise explanation with:\n"
                    f"1. key evidence\n2. task-by-task risk summary\n3. uncertainty/conflict note"
                ),
            },
        ]

        try:
            return self.llm_client.text_chat(self.model, messages, max_completion_tokens=700)
        except Exception:
            return fallback