from __future__ import annotations

from mm_rcna.policy.repair_policy import LearnedRepairPolicy, RuleRepairPolicy
from mm_rcna.schemas import RepairTraceStep


class OrchestratorRepairLoop:
    def __init__(self, config, llm_client=None, model: str | None = None) -> None:
        self.cfg = config
        self.rule_policy = RuleRepairPolicy()
        self.learned_policy = (
            LearnedRepairPolicy(config.repair.learned_policy_path)
            if config.repair.use_learned_policy
            else None
        )
        self.llm_client = llm_client
        self.model = model

    def _choose_action(self, verification, conflict, retrieval, pred, budget):
        used_llm = False

        if (
            self.llm_client is not None
            and getattr(self.llm_client, "ready", False)
            and self.model
            and getattr(budget, "api_budget_left", 0) > 0
        ):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Choose one repair action from this closed set only: "
                            "widen_interval, expand_standard_retrieval, "
                            "trigger_conflict_retrieval, rebuild_evidence, abstain.\n"
                            "Output strict JSON with keys: action, reason."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"violations={verification.violations}\n"
                            f"conflict_score={conflict.conflict_score}\n"
                            f"retrieval_std={retrieval.std}\n"
                            f"effective_n={retrieval.effective_n}\n"
                            f"interval_width={pred.interval_high - pred.interval_low}\n"
                        ),
                    },
                ]
                obj = self.llm_client.json_chat(self.model, messages, max_completion_tokens=250)
                action = str(obj.get("action", "")).strip()
                reason = str(obj.get("reason", "llm_selected")).strip()

                if action in {
                    "widen_interval",
                    "expand_standard_retrieval",
                    "trigger_conflict_retrieval",
                    "rebuild_evidence",
                    "abstain",
                }:
                    used_llm = True
                    return action, reason, used_llm
            except Exception:
                pass

        if self.learned_policy and self.learned_policy.ready:
            action, reason = self.learned_policy.choose_action(
                [
                    conflict.conflict_score,
                    retrieval.std,
                    retrieval.effective_n,
                    float(pred.interval_high - pred.interval_low),
                ]
            )
            return action, reason, used_llm

        action, reason = self.rule_policy.choose_action(verification.violations)
        return action, reason, used_llm

    def apply(
        self,
        round_idx,
        task,
        pred,
        verification,
        retrieval,
        conflict,
        conflict_summary,
        budget,
        fused_evidence,
        image_paths,
        study_id,
        subject_id,
        retrieval_agent,
        conflict_retrieval,
        mediator,
        verifier,
    ):
        action, reason, used_llm = self._choose_action(
            verification, conflict, retrieval, pred, budget
        )

        updated = []

        if action == "widen_interval":
            mid = pred.point
            half = max(
                (pred.interval_high - pred.interval_low) / 2.0,
                self.cfg.contracts.high_conflict_min_width / 2.0,
            )
            pred.interval_low = max(0.0, mid - 1.5 * half)
            pred.interval_high = min(1.0, mid + 1.5 * half)
            updated = ["task_prediction"]

        elif action == "expand_standard_retrieval":
            old_k = int(self.cfg.retrieval.top_k)
            new_k = min(max(old_k + 5, old_k * 2), old_k + 20)

            retrieval = retrieval_agent.retrieve(
                task.name,
                task.label_column,
                fused_evidence,
                image_paths,
                query_study_id=study_id,
                query_subject_id=subject_id,
                override_top_k=new_k,
            )
            pred.point = float((pred.point + retrieval.median) / 2.0)
            pred.interval_low = max(0.0, min(pred.interval_low, retrieval.q10))
            pred.interval_high = min(1.0, max(pred.interval_high, retrieval.q90))

            conflict = mediator.run(task.name, fused_evidence, retrieval, None)
            conflict_summary = (
                conflict_retrieval.retrieve(
                    task.name,
                    retrieval,
                    conflict,
                    label_column=task.label_column,
                    query_study_id=study_id,
                    query_subject_id=subject_id,
                )
                if conflict.trigger_conflict_retrieval
                else None
            )
            updated = ["retrieval", "task_prediction", "conflict"]

        elif action == "trigger_conflict_retrieval":
            if conflict_summary is None:
                conflict_summary = conflict_retrieval.retrieve(
                    task.name,
                    retrieval,
                    conflict,
                    label_column=task.label_column,
                    query_study_id=study_id,
                    query_subject_id=subject_id,
                )
            pred.point = float((pred.point + conflict_summary.median) / 2.0)
            pred.interval_low = max(0.0, min(pred.interval_low, conflict_summary.q10))
            pred.interval_high = min(1.0, max(pred.interval_high, conflict_summary.q90))
            updated = ["task_prediction", "conflict_retrieval"]

        elif action == "rebuild_evidence":
            retrieval = retrieval_agent.retrieve(
                task.name,
                task.label_column,
                fused_evidence,
                image_paths,
                query_study_id=study_id,
                query_subject_id=subject_id,
            )
            pred.point = float((pred.point + retrieval.median) / 2.0)
            updated = ["retrieval", "task_prediction"]

        elif action == "abstain":
            pred.abstain = True
            updated = ["task_prediction"]

        budget.rounds_left -= 1
        if used_llm:
            budget.api_budget_left = max(0, budget.api_budget_left - 1)

        trace_step = RepairTraceStep(
            round_idx=round_idx,
            action=action,
            reason=reason,
            updated_modules=updated,
        )
        return pred, trace_step, budget, retrieval, conflict, conflict_summary