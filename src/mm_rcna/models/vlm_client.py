from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def _extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}

    # Remove common markdown fences.
    cleaned = text
    cleaned = cleaned.replace("```json", "```")
    if "```" in cleaned:
        parts = cleaned.split("```")
        # Prefer the longest fenced or unfenced block that contains JSON braces.
        candidates = sorted(parts, key=len, reverse=True)
    else:
        candidates = [cleaned]

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue

        try:
            return json.loads(cand)
        except Exception:
            pass

        l = cand.find("{")
        r = cand.rfind("}")
        if l >= 0 and r > l:
            try:
                return json.loads(cand[l : r + 1])
            except Exception:
                pass

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}

def _fallback_items_from_text(text: str) -> List[Dict[str, Any]]:
    """Build conservative evidence items when LLaVA replies in free text instead of strict JSON."""
    text_low = str(text or "").lower()
    items: List[Dict[str, Any]] = []

    topic_rules = {
        "respiratory_status": [
            "opacity", "opacities", "effusion", "edema", "atelectasis",
            "consolidation", "infiltrate", "pneumonia", "pneumothorax",
            "hypoxia", "respiratory", "airspace",
        ],
        "criticality": [
            "critical", "severe", "unstable", "shock", "sepsis", "arrest",
            "icu", "decompensation", "distress",
        ],
        "hemodynamics": [
            "cardiomegaly", "congestion", "heart failure", "vascular",
            "hypotension", "tachycardia", "vasopressor",
        ],
        "support_devices": [
            "tube", "line", "catheter", "endotracheal", "ett",
            "intubated", "ventilator", "tracheostomy", "pacer",
        ],
    }

    for topic, kws in topic_rules.items():
        hits = [kw for kw in kws if kw in text_low]
        if not hits:
            continue

        if topic == "support_devices":
            tasks = ["icu_risk", "ventilation_risk"]
        elif topic in {"criticality", "hemodynamics"}:
            tasks = ["mortality_risk", "icu_risk"]
        else:
            tasks = ["mortality_risk", "icu_risk", "ventilation_risk"]

        items.append(
            {
                "topic": topic,
                "finding": "llava_free_text: " + ", ".join(hits[:5]),
                "score": min(0.85, 0.45 + 0.08 * len(hits)),
                "modality": "text+vision",
                "supports_tasks": tasks,
                "source": "llava_free_text",
            }
        )

    if not items and text.strip():
        items.append(
            {
                "topic": "respiratory_status",
                "finding": "llava_summary: " + text.strip().replace("\n", " ")[:180],
                "score": 0.50,
                "modality": "text+vision",
                "supports_tasks": ["mortality_risk", "icu_risk", "ventilation_risk"],
                "source": "llava_raw_summary",
            }
        )

    return items[:8]

class LlavaHFClient:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_4bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if load_4bit:
            kwargs["load_in_4bit"] = True

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.model.eval()

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _format_prompt(self, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            return self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
        except Exception:
            return f"USER: <image>\n{prompt}\nASSISTANT:"

     # 生成单张图片 evidence JSON
    def generate_json_for_image(
        self,
        image_path: str,
        report: str,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        task_names = task_names or ["mortality_risk", "icu_risk", "ventilation_risk"]
        prompt = f"""
You are a medical multimodal evidence extraction assistant.
Inspect the chest X-ray image and the provided radiology/clinical text.
Tasks: {', '.join(task_names)}
Clinical/radiology text: {str(report or '')[:3500]}
Return strict JSON only.
"""
        image = self._load_image(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature>0, temperature=max(self.temperature, 1e-6))
        decoded = self.processor.batch_decode(output[:, input_len:], skip_special_tokens=True)[0].strip()
        obj = _extract_json(decoded)
        obj["_raw_text"] = decoded[:2000]
        obj["_image_path"] = image_path
        items = obj.get("evidence_items", [])
        if not isinstance(items, list) or len(items) == 0:
            obj["evidence_items"] = _fallback_items_from_text(decoded)
            obj["summary"] = obj.get("summary") or decoded[:500]
            obj["_json_parse_failed_or_empty"] = True
        else:
            obj["_json_parse_failed_or_empty"] = False
        return obj

    # 多图片生成
    def generate_json_for_study(
        self,
        image_paths: List[str],
        report: str,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        all_items, summaries, raw_texts, errors = [], [], [], []
        for image_path in (image_paths or [])[:2]:
            try:
                obj = self.generate_json_for_image(image_path=image_path, report=report, task_names=task_names)
            except Exception as e:
                obj = {"evidence_items": [], "summary": f"llava_error: {repr(e)}", "_raw_text": "", "_error": repr(e)}
                errors.append(repr(e))
            items = obj.get("evidence_items", [])
            if isinstance(items, list):
                all_items.extend(items)
            summary = obj.get("summary", "")
            if summary:
                summaries.append(str(summary))
            raw_text = obj.get("_raw_text", "")
            if raw_text:
                raw_texts.append(str(raw_text))
        if not all_items and raw_texts:
            all_items.extend(_fallback_items_from_text(" ".join(raw_texts)))
        return {
            "evidence_items": all_items[:16],
            "summary": " | ".join(summaries)[:1200],
            "raw_text": " | ".join(raw_texts)[:2000],
            "errors": errors[:4],
            "num_images": len(image_paths or []),
        }