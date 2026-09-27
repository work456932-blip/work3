from __future__ import annotations

import re

from mm_rcna.schemas import AuditEvent, GovernanceOutput


class TextGovernanceAgent:
    def __init__(self, llm_client=None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model

    def _rule_clean(self, notes: str, report: str) -> GovernanceOutput:
        audit = []
        cleaned_notes = notes or ""
        cleaned_report = report or ""

        for token in ["name:", "mrn:", "dob:", "phone:"]:
            if token in cleaned_notes.lower() or token in cleaned_report.lower():
                audit.append(
                    AuditEvent(
                        name="pii_redaction_notice",
                        detail=f"Possible token detected: {token}",
                        severity="warning",
                    )
                )

        cleaned_notes = re.sub(r"\b\d{7,}\b", "[REDACTED_ID]", cleaned_notes)
        cleaned_report = re.sub(r"\b\d{7,}\b", "[REDACTED_ID]", cleaned_report)

        if not cleaned_report.strip():
            audit.append(
                AuditEvent(name="missing_report", detail="No report text found", severity="warning")
            )

        return GovernanceOutput(
            cleaned_notes=cleaned_notes,
            cleaned_report=cleaned_report,
            audit_events=audit,
        )

    def run(self, notes: str, report: str) -> GovernanceOutput:
        fallback = self._rule_clean(notes, report)

        if self.llm_client is None or not getattr(self.llm_client, "ready", False) or not self.model:
            return fallback

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical text-governance module. "
                    "Redact obvious identifiers, keep clinically useful information, "
                    "do not invent facts, and output strict JSON with keys: "
                    "cleaned_notes, cleaned_report, audit_events. "
                    "Each audit event must have keys: name, detail, severity."
                ),
            },
            {
                "role": "user",
                "content": f"NOTES:\n{notes or ''}\n\nREPORT:\n{report or ''}",
            },
        ]

        try:
            obj = self.llm_client.json_chat(self.model, messages, max_completion_tokens=900)

            events = []
            for ev in obj.get("audit_events", []):
                events.append(
                    AuditEvent(
                        name=str(ev.get("name", "llm_notice")),
                        detail=str(ev.get("detail", "")),
                        severity=str(ev.get("severity", "info")),
                    )
                )
            return GovernanceOutput(
                cleaned_notes=str(obj.get("cleaned_notes", fallback.cleaned_notes)),
                cleaned_report=str(obj.get("cleaned_report", fallback.cleaned_report)),
                audit_events=events if events else fallback.audit_events,
            )
        except Exception as e:
            fallback.audit_events.append(
                AuditEvent(
                    name="llm_governance_fallback",
                    detail=f"LLM governance failed: {e}",
                    severity="warning",
                )
            )
            return fallback