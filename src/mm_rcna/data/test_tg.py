from __future__ import annotations

from mm_rcna.config import load_config
from mm_rcna.data.text_governance import TextGovernanceAgent
from mm_rcna.models.api_clients import OpenAICompatibleClient


def print_result(test_name, out, llm_called):
    print(f"\n===== {test_name} =====")
    print(f"LLM called: {llm_called}")
    print(f"cleaned_notes: {out.cleaned_notes}")
    print(f"cleaned_report: {out.cleaned_report}")
    print("audit_events:")
    for ev in out.audit_events:
        print(f"  - name={ev.name}, severity={ev.severity}, detail={ev.detail}")


class DebugOpenAICompatibleClient(OpenAICompatibleClient):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.called = False

    def json_chat(self, model, messages, max_completion_tokens=900):
        self.called = True

        print("\n>>> Real LLM json_chat called")
        print("\n=== LLM INPUT MESSAGES ===")
        for m in messages:
            print(f"{m['role']}: {m['content']}")
        print("==========================\n")

        obj = super().json_chat(model, messages, max_completion_tokens=max_completion_tokens)

        print("=== LLM RAW RESPONSE ===")
        print(obj)
        print("========================\n")

        return obj


def test_real_llm_only():
    cfg = load_config("configs/default.yaml")

    client = DebugOpenAICompatibleClient(cfg.models.api)

    agent = TextGovernanceAgent(
        llm_client=client,
        model=cfg.models.api.llm_model,
    )

    notes = (
        "Name: Bob Smith. "
        "Phone: 1234567890. "
        "Patient ID: 12345678. "
        "The patient complains of mild shortness of breath for 2 days."
    )

    report = (
        "MRN: 87654321. "
        "DOB: 1990-01-01. "
        "Findings: No focal consolidation, pleural effusion, or pneumothorax. "
        "Heart size is within normal limits."
    )

    out = agent.run(notes, report)

    print_result("test_real_llm_only", out, llm_called=client.called)

    assert client.called is True
    assert isinstance(out.cleaned_notes, str)
    assert isinstance(out.cleaned_report, str)
    assert isinstance(out.audit_events, list)