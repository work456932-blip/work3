from __future__ import annotations

from mm_rcna.models.api_clients import NullAPIClient


def build_text_llm(name: str):
    return NullAPIClient(name)
