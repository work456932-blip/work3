from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from mm_rcna.config import load_config
from mm_rcna.models.text_encoder import TextEncoder
from mm_rcna.models.vision import VisionToolRunner
from mm_rcna.agents.evidence_builder import MultimodalEvidenceBuilder
from mm_rcna.agents.retrieval import RetrievalAgent, ConflictRetrievalAgent
from mm_rcna.agents.conflict import MediatorAgent
from mm_rcna.agents.task_coordinator import TaskCoordinatorAgent


class AttrDict(dict):
    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


def safe_get(obj, path: str, default=None):
    cur = obj
    for part in path.split("."):
        try:
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                cur = getattr(cur, part)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def save_json(obj, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_text(report_path: Path, notes_path: Path) -> tuple[str, str]:
    report = ""
    notes = ""

    if report_path.exists():
        report = report_path.read_text(encoding="utf-8", errors="ignore")

    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8", errors="ignore")

    return report, notes


def obj_to_dict(x):
    if isinstance(x, dict):
        return x
    if is_dataclass(x):
        return asdict(x)
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {"value": str(x)}


def load_conformal(path: str | None):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


class NullCalibrator:
    def __init__(self, conformal=None):
        self.conformal = conformal or {}

    def predict_interval(self, point, task_name=None):
        try:
            p = float(point)
        except Exception:
            p = 0.5
        width = 0.12
        lo = max(0.0, p - width)
        hi = min(1.0, p + width)
        return lo, hi


class OpenAICompatJSONClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        try:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self.ready = True
        except Exception:
            self.client = None
            self.ready = False

    @staticmethod
    def _extract_json(text: str):
        text = (text or "").strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            pass

        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(text[l : r + 1])
            except Exception:
                pass

        return {}

    def json_chat(self, model: str, messages, max_completion_tokens: int = 1200):
        if not self.ready or self.client is None:
            return {}

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_completion_tokens,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return self._extract_json(content)
        except Exception:
            return {}


def make_task(task_name: str):
    topic_map = {
        "mortality_risk": [
            "respiratory_status",
            "criticality",
            "hemodynamics",
            "support_devices",
        ],
        "icu_risk": [
            "respiratory_status",
            "criticality",
            "hemodynamics",
            "support_devices",
        ],
        "ventilation_risk": [
            "respiratory_status",
            "support_devices",
        ],
    }

    return AttrDict(
        name=task_name,
        evidence_topics=topic_map.get(task_name, []),
    )


def compute_coverage(fused_evidence, task) -> float:
    if not fused_evidence:
        return 0.0

    topics = set(task.evidence_topics or [])
    if not topics:
        rel = [
            x for x in fused_evidence
            if task.name in getattr(x, "supports_tasks", [])
        ]
        return float(min(1.0, len(rel) / 4.0))

    hit_topics = set(
        x.topic
        for x in fused_evidence
        if task.name in getattr(x, "supports_tasks", [])
        and x.topic in topics
    )

    return float(len(hit_topics) / max(1, len(topics)))


def build_vlm_client(args, cfg):
    if args.agent_backbone != "llava":
        return None

    from mm_rcna.models.vlm_client import LlavaHFClient

    dtype = safe_get(cfg, "vlm.dtype", "bfloat16")
    load_4bit = bool(safe_get(cfg, "vlm.load_4bit", False))
    max_new_tokens = int(safe_get(cfg, "vlm.max_new_tokens", 512))
    temperature = float(safe_get(cfg, "vlm.temperature", 0.0))

    return LlavaHFClient(
        model_name=args.vlm_model,
        device=args.device,
        dtype=dtype,
        load_4bit=load_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def build_llm_client(args, cfg):
    if args.agent_backbone != "qwen":
        return None, None

    try:
        if args.llm_model is not None:
            cfg.models.api.llm_model = args.llm_model
    except Exception:
        pass

    try:
        if getattr(cfg.models.api, "enabled", False):
            client = OpenAICompatJSONClient(
                base_url=cfg.models.api.base_url,
                api_key=cfg.models.api.api_key,
                timeout=float(getattr(cfg.models.api, "timeout", 120)),
            )
            return client, getattr(cfg.models.api, "llm_model", args.llm_model)
    except Exception:
        pass

    return None, args.llm_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="./artifacts/pipeline_output.json")
    parser.add_argument("--conformal-json", default="./artifacts/conformal.json")

    parser.add_argument(
        "--agent-backbone",
        choices=["qwen", "llava"],
        default="qwen",
        help="Independent evidence-generation backbone.",
    )

    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--vlm-model",
        default="llava-hf/llava-1.5-7b-hf",
        help="LLaVA model name when --agent-backbone llava.",
    )

    parser.add_argument("--cache-dir", default="./artifacts/embedding_cache")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--mirror-reports-dir", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    study_id = args.study_id

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(cfg.data.reports_dir) / f"{study_id}.txt"
    notes_path = Path(cfg.data.notes_dir) / f"{study_id}.txt"
    image_paths = sorted(
        [str(p) for p in Path(cfg.data.images_dir).glob(f"{study_id}_*")]
    )

    report, notes = load_text(report_path, notes_path)
    merged_text = (report + "\n" + notes).strip()

    study = AttrDict(
        study_id=study_id,
        subject_id="",
        report=report,
        notes=notes,
        image_paths=image_paths,
    )

    gov = AttrDict(
        cleaned_report=report,
        cleaned_notes=notes,
        audit_flags=[],
    )

    cache_key = stable_hash(
        {
            "study_id": study_id,
            "image_paths": image_paths,
            "report_exists": report_path.exists(),
            "notes_exists": notes_path.exists(),
            "text_encoder": getattr(cfg.models, "text_encoder_name", ""),
            "image_encoder": getattr(cfg.models, "image_encoder_name", ""),
            "vision_backbone": getattr(cfg.models, "vision_backbone_name", ""),
            "image_size": getattr(cfg.models, "image_size", 512),
            "agent_backbone": args.agent_backbone,
            "llm_model": args.llm_model or "",
            "vlm_model": args.vlm_model or "",
        }
    )
    cache_file = cache_dir / f"{study_id}_{cache_key}.json"

    if args.reuse_embeddings and cache_file.exists():
        cached = load_json(cache_file)
        vision_out = AttrDict(cached["vision_out"])
        text_vec = np.asarray(cached["text_vec"], dtype=np.float32)
        used_cached_embeddings = True
    else:
        text_encoder = TextEncoder(cfg.models.text_encoder_name, device=args.device)
        vision = VisionToolRunner(
            cfg.models.image_encoder_name,
            cfg.models.vision_backbone_name,
            cfg.models.lesion_labels,
            cfg.models.lung_regions,
            image_size=cfg.models.image_size,
            device=args.device,
            checkpoint_path=cfg.models.vision_checkpoint
            if cfg.models.use_trained_vision_heads
            else None,
        )

        text_vec = text_encoder.encode([merged_text])[0].astype(np.float32)
        vout = vision.run(image_paths)

        vision_out = AttrDict(
            {
                "feature_vector": list(getattr(vout, "feature_vector", [])),
                "lesion_scores": dict(getattr(vout, "lesion_scores", {})),
                "region_scores": dict(getattr(vout, "region_scores", {})),
                "quality_flags": list(getattr(vout, "quality_flags", [])),
            }
        )

        save_json(
            {
                "study_id": study_id,
                "text_vec": text_vec.tolist(),
                "vision_out": dict(vision_out),
            },
            cache_file,
        )
        used_cached_embeddings = False

    llm_client, llm_model = build_llm_client(args, cfg)
    vlm_client = build_vlm_client(args, cfg)

    evidence_builder = MultimodalEvidenceBuilder(
        cfg,
        llm_client=llm_client,
        vlm_client=vlm_client,
        model=llm_model if args.agent_backbone == "qwen" else args.vlm_model,
        agent_backbone=args.agent_backbone,
    )

    retrieval_agent = RetrievalAgent(cfg, device=args.device)
    conflict_retrieval_agent = ConflictRetrievalAgent(cfg)
    mediator_agent = MediatorAgent()
    calibrator = NullCalibrator(load_conformal(args.conformal_json))
    task_coordinator = TaskCoordinatorAgent(cfg, calibrator)

    fused_evidence = evidence_builder.run(
        study=study,
        gov=gov,
        vision_out=vision_out,
        image_paths=image_paths,
    )

    tasks = ["mortality_risk", "icu_risk", "ventilation_risk"]

    predictions = []
    verifications = {}
    retrievals = {}
    conflicts = {}
    conflict_retrievals = {}

    for task_name in tasks:
        task = make_task(task_name)
        label_column = task_name

        retrieval = retrieval_agent.retrieve(
            task_name=task_name,
            label_column=label_column,
            evidence=fused_evidence,
            image_paths=image_paths,
            query_study_id=study_id,
            query_subject_id=study.subject_id,
        )
        retrievals[task_name] = obj_to_dict(retrieval)

        conflict = mediator_agent.run(
            task_name=task_name,
            fused_evidence=fused_evidence,
            retrieval=retrieval,
            vision_out=vision_out,
        )
        conflicts[task_name] = obj_to_dict(conflict)

        conflict_summary = None
        if getattr(conflict, "trigger_conflict_retrieval", False):
            try:
                conflict_summary = conflict_retrieval_agent.retrieve(
                    task_name=task_name,
                    retrieval_summary=retrieval,
                    conflict=conflict,
                    label_column=label_column,
                    query_study_id=study_id,
                    query_subject_id=study.subject_id,
                )
                conflict_retrievals[task_name] = obj_to_dict(conflict_summary)
            except Exception:
                conflict_summary = None

        coverage = compute_coverage(fused_evidence, task)

        pred = task_coordinator.predict_one(
            task=task,
            fused_evidence=fused_evidence,
            coverage=coverage,
            retrieval=retrieval,
            conflict=conflict,
            conflict_summary=conflict_summary,
        )

        pred_dict = obj_to_dict(pred)
        predictions.append(pred_dict)

        verifications[task_name] = {
            "task_name": task_name,
            "passed": not bool(pred_dict.get("abstain", False)),
            "violations": []
            if not bool(pred_dict.get("abstain", False))
            else ["abstained"],
            "confidence": float(pred_dict.get("point", 0.5)),
        }

    result = {
        "study_id": study_id,
        "agent_backbone": args.agent_backbone,
        "llm_model": args.llm_model,
        "vlm_model": args.vlm_model if args.agent_backbone == "llava" else None,
        "predictions": predictions,
        "verifications": verifications,
        "retrievals": retrievals,
        "conflicts": conflicts,
        "cache_file": str(cache_file),
        "cache_dir": str(cache_dir),
        "used_cached_embeddings": used_cached_embeddings,
    }

    if conflict_retrievals:
        result["conflict_retrievals"] = conflict_retrievals

    save_json(result, args.output_json)

    if args.mirror_reports_dir:
        mirror_path = Path(args.mirror_reports_dir) / f"{study_id}.json"
        save_json(result, mirror_path)
        print(f"Mirrored to {mirror_path}")

    print(f"Saved to {args.output_json}")
    print(f"cache_dir={cache_dir}")
    print(f"used_cached_embeddings={used_cached_embeddings}")


if __name__ == "__main__":
    main()

