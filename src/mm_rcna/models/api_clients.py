from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from mm_rcna.config import APIConfig


class NullAPIClient:
    def __init__(self, name: str = "null-client") -> None:
        self.name = name
        self.ready = False

    def text_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: int = 800,
    ) -> str:
        raise RuntimeError("NullAPIClient cannot call text_chat")

    def json_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: int = 800,
    ) -> Dict[str, Any]:
        raise RuntimeError("NullAPIClient cannot call json_chat")

    def embeddings(self, model: str, input_texts: List[str]) -> Dict[str, Any]:
        raise RuntimeError("NullAPIClient cannot call embeddings")


class OpenAICompatibleClient:
    def __init__(self, cfg: APIConfig):
        self.cfg = cfg
        self.base_url = self._normalize_base_url(cfg.base_url)
        self.auth_token = self._resolve_auth_token(cfg)
        self.ready = bool(self.base_url)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base_url = (base_url or "").strip().rstrip("/")
        for suffix in ("/embeddings", "/chat/completions", "/responses"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
        return base_url

    @staticmethod
    def _resolve_auth_token(cfg: APIConfig) -> str:
        if cfg.api_key:
            return cfg.api_key
        if cfg.auth_token_env:
            return os.getenv(cfg.auth_token_env, "")
        return ""

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.cfg.extra_headers:
            headers.update(self.cfg.extra_headers)
        return headers

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError("API client not ready: missing base_url.")

        url = f"{self.base_url}/{route.lstrip('/')}"
        last_err: Optional[Exception] = None

        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_sleep_seconds)
                else:
                    raise

        raise RuntimeError(f"Unreachable retry state, last_err={last_err}")

    @staticmethod
    def _extract_text(obj: Dict[str, Any]) -> str:
        try:
            return obj["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Cannot parse chat completion response: {e}; obj={obj}")

    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        # 尝试从大段文本中截 JSON
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            return json.loads(candidate)

        raise RuntimeError(f"Model did not return valid JSON. Raw content:\n{text}")

    def text_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: int = 800,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        obj = self._post("chat/completions", payload)
        return self._extract_text(obj)

    def json_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_completion_tokens: int = 800,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_completion_tokens": max_completion_tokens,
        }

        # 优先尝试原生 JSON 模式
        if self.cfg.supports_json_response_format:
            payload_with_json = dict(payload)
            payload_with_json["response_format"] = {"type": "json_object"}
            try:
                obj = self._post("chat/completions", payload_with_json)
                content = self._extract_text(obj)
                return self._extract_json_from_text(content)
            except Exception:
                pass

        # fallback：强约束提示词 + 普通 text completion
        hardened_messages = list(messages)
        if hardened_messages:
            hardened_messages = hardened_messages.copy()
            hardened_messages[0] = {
                "role": hardened_messages[0]["role"],
                "content": (
                    str(hardened_messages[0]["content"]).strip()
                    + "\n\nReturn ONLY valid JSON. No markdown. No explanation."
                ),
            }
        else:
            hardened_messages = [
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON. No markdown. No explanation.",
                }
            ]

        payload["messages"] = hardened_messages
        obj = self._post("chat/completions", payload)
        content = self._extract_text(obj)
        return self._extract_json_from_text(content)

    def embeddings(self, model: str, input_texts: List[str]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "input": input_texts,
        }
        return self._post("embeddings", payload)