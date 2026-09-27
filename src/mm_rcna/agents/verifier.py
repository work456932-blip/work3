from __future__ import annotations

from mm_rcna.schemas import VerificationResult


class ContractVerifierAgent:
    def __init__(self, config, llm_client=None, model: str | None = None) -> None:
        self.cfg = config
        self.llm_client = llm_client
        self.model = model

    def verify_one(self, pred, retrieval, conflict, gov, vision_out):
        violations = []

        width = float(pred.interval_high - pred.interval_low)
        if pred.interval_low < self.cfg.contracts.min_interval_low or pred.interval_high > self.cfg.contracts.max_interval_high:
            violations.append("interval_out_of_bounds")

        if retrieval.effective_n < self.cfg.retrieval.min_effective_n:
            violations.append("low_retrieval_support")

        if retrieval.unstable_distribution:
            violations.append("unstable_retrieval_distribution")

        if conflict.conflict_score >= self.cfg.contracts.high_conflict_threshold and width < self.cfg.contracts.high_conflict_min_width:
            violations.append("interval_too_narrow_for_high_conflict")

        if not (gov.cleaned_report or "").strip():
            violations.append("missing_report_text")

        passed = len(violations) == 0
        confidence = max(0.0, min(1.0, 1.0 - conflict.conflict_score - 0.2 * len(violations)))

        if self.llm_client is not None and getattr(self.llm_client, "ready", False) and self.model:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You review the current violations conservatively. "
                            "Output strict JSON with keys: violations, confidence_adjustment. "
                            "Do not remove core rule-based safety violations unless clearly unsupported."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"task={pred.task_name}\n"
                            f"point={pred.point}\n"
                            f"interval=[{pred.interval_low},{pred.interval_high}]\n"
                            f"retrieval_effective_n={retrieval.effective_n}\n"
                            f"retrieval_unstable={retrieval.unstable_distribution}\n"
                            f"conflict_score={conflict.conflict_score}\n"
                            f"default_violations={violations}"
                        ),
                    },
                ]
                obj = self.llm_client.json_chat(self.model, messages, max_completion_tokens=300)
                llm_violations = obj.get("violations", [])
                if isinstance(llm_violations, list) and llm_violations:
                    violations = list(dict.fromkeys([str(x) for x in llm_violations]))
                    passed = len(violations) == 0
                confidence = max(
                    0.0,
                    min(1.0, confidence + float(obj.get("confidence_adjustment", 0.0))),
                )
            except Exception:
                pass

        return VerificationResult(
            task_name=pred.task_name,
            passed=passed,
            violations=violations,
            confidence=confidence,
        )