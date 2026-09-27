from __future__ import annotations

import numpy as np
from mm_rcna.schemas import TaskPrediction


class TaskCoordinatorAgent:
    def __init__(self, config, calibrator) -> None:
        self.cfg = config
        self.calibrator = calibrator

    def _task_weights(self, task_name: str):
        """
        这一版目标：
        1) 不再把 point 压得过低
        2) 保住 Acc，同时尽量抬 AUROC
        3) 三个任务显式区分
        """
        if task_name == "mortality_risk":
            return {
                "evidence": 0.22,
                "vision": 0.10,
                "retrieval_pos": 0.30,
                "top1_label": 0.14,
                "top1_score": 0.08,
                "coverage": 0.08,
                "support_bonus": 0.06,
                "conflict_penalty": 0.10,
                "instability_penalty": 0.02,
                "insufficient_penalty": 0.02,
                "offset": 0.08,
                "scale": 0.92,
            }

        if task_name == "icu_risk":
            return {
                "evidence": 0.22,
                "vision": 0.08,
                "retrieval_pos": 0.34,
                "top1_label": 0.16,
                "top1_score": 0.08,
                "coverage": 0.08,
                "support_bonus": 0.06,
                "conflict_penalty": 0.08,
                "instability_penalty": 0.02,
                "insufficient_penalty": 0.02,
                "offset": 0.06,
                "scale": 0.94,
            }

        if task_name == "ventilation_risk":
            return {
                "evidence": 0.14,
                "vision": 0.22,
                "retrieval_pos": 0.28,
                "top1_label": 0.16,
                "top1_score": 0.10,
                "coverage": 0.06,
                "support_bonus": 0.04,
                "conflict_penalty": 0.07,
                "instability_penalty": 0.015,
                "insufficient_penalty": 0.02,
                "offset": 0.08,
                "scale": 0.92,
            }

        return {
            "evidence": 0.22,
            "vision": 0.10,
            "retrieval_pos": 0.30,
            "top1_label": 0.14,
            "top1_score": 0.08,
            "coverage": 0.08,
            "support_bonus": 0.05,
            "conflict_penalty": 0.08,
            "instability_penalty": 0.02,
            "insufficient_penalty": 0.02,
            "offset": 0.07,
            "scale": 0.93,
        }

    def _safe_mean(self, xs):
        xs = [float(x) for x in xs]
        return float(np.mean(xs)) if xs else 0.0

    def _safe_max(self, xs):
        xs = [float(x) for x in xs]
        return float(np.max(xs)) if xs else 0.0

    def predict_one(self, task, fused_evidence, coverage, retrieval, conflict, conflict_summary=None):
        task_items = [
            x for x in fused_evidence
            if task.name in x.supports_tasks and (not task.evidence_topics or x.topic in task.evidence_topics)
        ]

        evidence_score = self._safe_mean([x.score for x in task_items])
        evidence_top = self._safe_max([x.score for x in task_items])

        vision_items = [x for x in task_items if getattr(x, "modality", "") == "vision"]
        text_items = [x for x in task_items if getattr(x, "modality", "") == "text"]

        vision_score = self._safe_mean([x.score for x in vision_items])
        text_score = self._safe_mean([x.score for x in text_items]) if text_items else evidence_score

        neighbors = getattr(retrieval, "neighbors", []) or []
        retrieval_pos = float(getattr(retrieval, "median", 0.0))

        top1_label = float(neighbors[0].label_value) if neighbors else retrieval_pos
        top1_score = float(neighbors[0].score) if neighbors else 0.0

        top3 = neighbors[:3]
        if top3:
            top3_labels = np.asarray([float(x.label_value) for x in top3], dtype=np.float32)
            top3_scores = np.asarray([float(x.score) for x in top3], dtype=np.float32)

            # 更平滑，别让 top1 完全统治
            top3_scores = np.exp((top3_scores - top3_scores.max()) / 0.05)
            top3_scores = top3_scores / max(float(top3_scores.sum()), 1e-6)
            top3_consensus = float(np.sum(top3_scores * top3_labels))
        else:
            top3_consensus = retrieval_pos

        # 支持数奖励：有上限，避免过头
        support_bonus = min(0.05, 0.006 * float(getattr(retrieval, "effective_n", 0)))

        w = self._task_weights(task.name)

        # 基础融合
        base_point = (
            w["evidence"] * evidence_score
            + w["vision"] * vision_score
            + w["retrieval_pos"] * retrieval_pos
            + w["top1_label"] * top3_consensus
            + w["top1_score"] * top1_score
            + w["coverage"] * float(coverage)
            + w["support_bonus"] * support_bonus
        )

        # evidence_top 单独给一点增益，帮助排序
        base_point += 0.06 * evidence_top

        # 文本和视觉相近，说明一致性较好，轻微奖励
        cross_modal_gap = abs(text_score - vision_score)
        if cross_modal_gap < 0.10:
            base_point += 0.015
        elif cross_modal_gap > 0.30:
            base_point -= 0.015

        # conflict retrieval 只做小修正，不主导
        if conflict_summary is not None and not getattr(conflict_summary, "insufficient_support", True):
            base_point = 0.90 * base_point + 0.10 * float(conflict_summary.median)

        # 惩罚项减弱，避免 point 被整体压塌
        point = base_point - w["conflict_penalty"] * float(conflict.conflict_score)

        if getattr(retrieval, "unstable_distribution", False):
            point -= w["instability_penalty"]

        if getattr(retrieval, "insufficient_support", False):
            point -= w["insufficient_penalty"]

        # 关键：把分数整体往中间抬一点，避免最优阈值掉到 0.05~0.07
        point = w["offset"] + w["scale"] * point

        point = float(np.clip(point, 0.0, 1.0))
        lo, hi = self.calibrator.predict_interval(point)

        if float(conflict.conflict_score) >= float(self.cfg.contracts.high_conflict_threshold):
            extra = 0.05 + 0.08 * float(conflict.conflict_score)
            lo = max(float(self.cfg.contracts.min_interval_low), float(lo - extra))
            hi = min(float(self.cfg.contracts.max_interval_high), float(hi + extra))

        abstain = False
        if getattr(retrieval, "insufficient_support", False) and float(conflict.conflict_score) > 0.86:
            abstain = True
        if (hi - lo) > 0.86:
            abstain = True

        rationale = (
            f"task={task.name}, evidence={evidence_score:.3f}, evidence_top={evidence_top:.3f}, "
            f"vision={vision_score:.3f}, retrieval={retrieval_pos:.3f}, "
            f"top1_label={top1_label:.3f}, top1_score={top1_score:.3f}, "
            f"coverage={float(coverage):.3f}, support_n={retrieval.effective_n}, "
            f"conflict={conflict.conflict_score:.3f}"
        )

        return TaskPrediction(
            task_name=task.name,
            point=point,
            interval_low=float(lo),
            interval_high=float(hi),
            abstain=abstain,
            rationale=rationale,
        )


# backward-compatible alias
TaskCoordinator = TaskCoordinatorAgent