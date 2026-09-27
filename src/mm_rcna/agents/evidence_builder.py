from __future__ import annotations

import json
from typing import List

from mm_rcna.config import AppConfig
from mm_rcna.schemas import EvidenceItem, GovernanceOutput, StudyRecord, VisionOutput


TOPIC_KEYWORDS = {
    "respiratory_status": [
        "opacity",
        "effusion",
        "pneumothorax",
        "respiratory",
        "oxygen",
        "ventilation",
        "edema",
        "atelectasis",
        "infiltrate",
        "consolidation",
        "pneumonia",
        "hypoxia",
    ],
    "criticality": [
        "critical",
        "unstable",
        "shock",
        "icu",
        "decompensation",
        "severe",
        "sepsis",
        "arrest",
    ],
    "hemodynamics": [
        "hypotension",
        "vasopressor",
        "tachycardia",
        "hemodynamic",
        "cardiomegaly",
        "congestion",
        "heart failure",
    ],
    "support_devices": [
        "tube",
        "line",
        "catheter",
        "intubated",
        "intubation",
        "endotracheal",
        "ventilator",
        "pacer",
        "tracheostomy",
        "ett",
        "support device",
    ],
}

ALLOWED_TOPICS = [
    "respiratory_status",
    "criticality",
    "hemodynamics",
    "support_devices",
]

DEFAULT_TASKS = [
    "mortality_risk",
    "icu_risk",
    "ventilation_risk",
]


class MultimodalEvidenceBuilder:
    def __init__(
        self,
        config: AppConfig,
        llm_client=None,
        vlm_client=None,
        model: str | None = None,
        agent_backbone: str = "qwen",
    ) -> None:
        self.cfg = config
        self.llm_client = llm_client
        self.vlm_client = vlm_client
        self.model = model
        self.agent_backbone = agent_backbone

    def _text_evidence_rule(self, text: str) -> List[EvidenceItem]:
        text_low = (text or "").lower()
        items: List[EvidenceItem] = []

        for topic, kws in TOPIC_KEYWORDS.items():
            hits = [kw for kw in kws if kw in text_low]
            if not hits:
                continue

            if topic == "support_devices":
                tasks = ["icu_risk", "ventilation_risk"]
            elif topic == "criticality":
                tasks = ["mortality_risk", "icu_risk"]
            else:
                tasks = DEFAULT_TASKS

            items.append(
                EvidenceItem(
                    topic=topic,
                    finding=", ".join(hits[:4]),
                    score=min(1.0, 0.18 * len(hits) + 0.22),
                    modality="text",
                    supports_tasks=tasks,
                    source="rule_text",
                )
            )

        return items

    def _vision_topic_from_lesion(self, lesion: str) -> str:
        lesion_low = (lesion or "").lower()

        if any(x in lesion_low for x in ["tube", "line", "catheter", "device", "pacer", "support"]):
            return "support_devices"
        if any(x in lesion_low for x in ["shock", "cardiomegaly", "congestion", "heart"]):
            return "hemodynamics"
        return "respiratory_status"

    def _vision_evidence_rule(self, vision_out: VisionOutput) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []

        lesion_scores = getattr(vision_out, "lesion_scores", {}) or {}
        for lesion, score in lesion_scores.items():
            try:
                score = float(score)
            except Exception:
                continue

            if score < 0.35:
                continue

            topic = self._vision_topic_from_lesion(lesion)

            if topic == "support_devices":
                tasks = ["icu_risk", "ventilation_risk"]
            elif topic == "hemodynamics":
                tasks = ["mortality_risk", "icu_risk"]
            else:
                tasks = DEFAULT_TASKS

            items.append(
                EvidenceItem(
                    topic=topic,
                    finding=str(lesion),
                    score=score,
                    modality="vision",
                    supports_tasks=tasks,
                    source="vision_tool",
                )
            )

        return items

    def _build_llm_payload(
        self,
        study: StudyRecord,
        gov: GovernanceOutput,
        vision_out: VisionOutput,
    ) -> dict:
        return {
            "study_id": study.study_id,
            "subject_id": getattr(study, "subject_id", None),
            "cleaned_notes": getattr(gov, "cleaned_notes", "") or "",
            "cleaned_report": getattr(gov, "cleaned_report", "") or "",
            "vision_summary": {
                "lesion_scores": getattr(vision_out, "lesion_scores", {}) or {},
                "region_scores": getattr(vision_out, "region_scores", {}) or {},
                "quality_flags": getattr(vision_out, "quality_flags", []) or [],
            },
            "allowed_topics": ALLOWED_TOPICS,
            "allowed_tasks": DEFAULT_TASKS,
            "requirements": {
                "max_items": 8,
                "score_range": [0.0, 1.0],
                "be_conservative": True,
                "no_final_risk_prediction": True,
            },
        }

    def _llm_multimodal_evidence(
        self,
        study: StudyRecord,
        gov: GovernanceOutput,
        vision_out: VisionOutput,
    ) -> List[EvidenceItem]:
        if self.llm_client is None or not getattr(self.llm_client, "ready", False) or not self.model:
            return []

        payload = self._build_llm_payload(study, gov, vision_out)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical multimodal evidence extraction module.\n"
                    "Your job is to extract evidence items from cleaned text and vision summary.\n"
                    "Do NOT output final risk predictions.\n"
                    "Output strict JSON with top-level key `evidence_items`.\n"
                    "Each evidence item must contain:\n"
                    "- topic: one of respiratory_status, criticality, hemodynamics, support_devices\n"
                    "- finding: short text\n"
                    "- score: float in [0,1]\n"
                    "- modality: one of text, vision, text+vision\n"
                    "- supports_tasks: subset of mortality_risk, icu_risk, ventilation_risk\n"
                    "- source: short snake_case string\n"
                    "Be conservative. Prefer fewer, higher-quality items."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

        try:
            max_tokens = int(getattr(self.cfg.models.api, "max_completion_tokens", 1200))
            obj = self.llm_client.json_chat(
                self.model,
                messages,
                max_completion_tokens=max_tokens,
            )
        except Exception:
            return []

        raw_items = obj.get("evidence_items", [])
        if not isinstance(raw_items, list):
            return []

        out: List[EvidenceItem] = []

        for x in raw_items[:8]:
            try:
                topic = str(x.get("topic", "respiratory_status")).strip()
                if topic not in ALLOWED_TOPICS:
                    continue

                supports_tasks = x.get("supports_tasks", DEFAULT_TASKS)
                if not isinstance(supports_tasks, list):
                    supports_tasks = DEFAULT_TASKS
                supports_tasks = [str(t) for t in supports_tasks if str(t) in DEFAULT_TASKS]
                if not supports_tasks:
                    supports_tasks = DEFAULT_TASKS

                score = float(x.get("score", 0.5))
                score = max(0.0, min(1.0, score))

                modality = str(x.get("modality", "text")).strip().lower()
                if modality not in {"text", "vision", "text+vision"}:
                    modality = "text"

                finding = str(x.get("finding", "")).strip()
                if not finding:
                    continue

                out.append(
                    EvidenceItem(
                        topic=topic,
                        finding=finding[:240],
                        score=score,
                        modality=modality,
                        supports_tasks=supports_tasks,
                        source=str(x.get("source", "llm_multimodal")),
                    )
                )
            except Exception:
                continue

        return out

    def _vlm_evidence(
        self,
        study: StudyRecord,
        gov: GovernanceOutput,
        image_paths=None,
    ) -> List[EvidenceItem]:
        if self.vlm_client is None:
            print("[VLM] skipped: vlm_client is None")
            return []

        merged_text = (
            str(getattr(gov, "cleaned_notes", "") or "")
            + "\n\n"
            + str(getattr(gov, "cleaned_report", "") or "")
        )

        try:
            obj = self.vlm_client.generate_json_for_study(
                image_paths=image_paths or [],
                report=merged_text,
                task_names=DEFAULT_TASKS,
            )
        except Exception as e:
            print(f"[VLM] generate_json_for_study failed: {repr(e)}")
            return [
                EvidenceItem(
                    topic="respiratory_status",
                    finding=f"llava_call_failed: {repr(e)[:180]}",
                    score=0.50,
                    modality="text+vision",
                    supports_tasks=DEFAULT_TASKS,
                    source="llava_error",
                )
            ]

        raw_items = obj.get("evidence_items", [])
        raw_text = str(obj.get("raw_text", "") or "")
        summary = str(obj.get("summary", "") or "")
        errors = obj.get("errors", [])
        num_images = obj.get("num_images", len(image_paths or []))

        print(
            f"[VLM] study={getattr(study, 'study_id', '')} "
            f"images={num_images} "
            f"raw_items={len(raw_items) if isinstance(raw_items, list) else 'NA'} "
            f"summary_len={len(summary)} "
            f"raw_len={len(raw_text)} "
            f"errors={errors[:1] if isinstance(errors, list) else errors}"
        )

        if not isinstance(raw_items, list):
            raw_items = []

        out: List[EvidenceItem] = []

        for x in raw_items[:16]:
            if not isinstance(x, dict):
                continue

            topic = str(x.get("topic", "respiratory_status")).strip()
            if topic not in ALLOWED_TOPICS:
                topic = "respiratory_status"

            finding = str(x.get("finding", "")).strip()
            if not finding:
                continue

            modality = str(x.get("modality", "text+vision")).strip().lower()
            if modality not in {"text", "vision", "text+vision"}:
                modality = "text+vision"

            try:
                score = float(x.get("score", 0.5))
            except Exception:
                score = 0.5
            score = max(0.0, min(1.0, score))

            supports_tasks = x.get("supports_tasks", DEFAULT_TASKS)
            if not isinstance(supports_tasks, list):
                supports_tasks = DEFAULT_TASKS

            supports_tasks = [
                str(t) for t in supports_tasks
                if str(t) in DEFAULT_TASKS
            ]
            if not supports_tasks:
                supports_tasks = DEFAULT_TASKS

            out.append(
                EvidenceItem(
                    topic=topic,
                    finding=("llava13b: " + finding)[:240],
                    score=score,
                    modality=modality,
                    supports_tasks=supports_tasks,
                    source=str(x.get("source", "llava13b")),
                )
            )

        if not out:
            fallback_text = summary or raw_text
            if fallback_text:
                out.append(
                    EvidenceItem(
                        topic="respiratory_status",
                        finding=("llava13b_summary: " + fallback_text.replace("\n", " "))[:240],
                        score=0.50,
                        modality="text+vision",
                        supports_tasks=DEFAULT_TASKS,
                        source="llava13b_raw_summary",
                    )
                )
            else:
                out.append(
                    EvidenceItem(
                        topic="respiratory_status",
                        finding="llava13b_no_parseable_output",
                        score=0.50,
                        modality="text+vision",
                        supports_tasks=DEFAULT_TASKS,
                        source="llava13b_empty",
                    )
                )

        return out

    @staticmethod
    def _dedup(items: List[EvidenceItem]) -> List[EvidenceItem]:
        seen = set()
        out = []

        for item in items:
            key = (
                item.topic.strip().lower(),
                item.finding.strip().lower(),
                item.modality.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)

        return out

    def run(
        self,
        study: StudyRecord,
        gov: GovernanceOutput,
        vision_out: VisionOutput,
        image_paths=None,
    ) -> List[EvidenceItem]:
        merged_text = (
            str(getattr(gov, "cleaned_notes", "") or "")
            + "\n\n"
            + str(getattr(gov, "cleaned_report", "") or "")
        )

        if self.agent_backbone == "qwen":
            llm_items = self._llm_multimodal_evidence(study, gov, vision_out)
            text_items = self._text_evidence_rule(merged_text)
            vision_items = self._vision_evidence_rule(vision_out)

            if llm_items:
                return self._dedup(llm_items + vision_items[:4] + text_items[:4])

            return self._dedup(text_items + vision_items)
        
        if self.agent_backbone == "llava":
            vlm_items = self._vlm_evidence(
                study=study,
                gov=gov,
                image_paths=image_paths,
            )
            text_items = self._text_evidence_rule(merged_text)
            vision_items = self._vision_evidence_rule(vision_out)

            print(
                f"[EVIDENCE] backbone=llava "
                f"vlm_items={len(vlm_items)} "
                f"text_items={len(text_items)} "
                f"vision_items={len(vision_items)}"
            )

            return self._dedup(vlm_items + vision_items[:4] + text_items[:4])

        raise ValueError(f"Unknown agent_backbone: {self.agent_backbone}")