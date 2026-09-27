from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


class Dumpable:
    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class AuditEvent(Dumpable):
    name: str
    detail: str
    severity: str = "info"


@dataclass
class ImageRef(Dumpable):
    path: str
    view: str = "unknown"


@dataclass
class StudyRecord(Dumpable):
    study_id: str
    subject_id: Optional[str]
    notes: str
    report: str
    images: List[ImageRef]


@dataclass
class GovernanceOutput(Dumpable):
    cleaned_notes: str
    cleaned_report: str
    audit_events: List[AuditEvent] = field(default_factory=list)


@dataclass
class VisionOutput(Dumpable):
    feature_vector: List[float]
    lesion_scores: Dict[str, float]
    region_scores: Dict[str, float]
    quality_flags: List[str] = field(default_factory=list)


@dataclass
class EvidenceItem(Dumpable):
    topic: str
    finding: str
    score: float
    modality: str
    supports_tasks: List[str]
    source: str


@dataclass
class RetrievalNeighbor(Dumpable):
    study_id: str
    score: float
    match_reasons: List[str]
    label_value: float


@dataclass
class RetrievalSummary(Dumpable):
    task_name: str
    neighbors: List[RetrievalNeighbor]
    median: float
    q10: float
    q90: float
    std: float
    effective_n: int
    insufficient_support: bool
    unstable_distribution: bool
    query_text: str


@dataclass
class ConflictReport(Dumpable):
    task_name: str
    conflict_score: float
    conflict_sources: List[str]
    conflict_vector: List[float]
    trigger_conflict_retrieval: bool


@dataclass
class TaskPrediction(Dumpable):
    task_name: str
    point: float
    interval_low: float
    interval_high: float
    abstain: bool = False
    rationale: str = ""


@dataclass
class VerificationResult(Dumpable):
    task_name: str
    passed: bool
    violations: List[str]
    confidence: float


@dataclass
class RepairBudget(Dumpable):
    rounds_left: int
    api_budget_left: int
    token_budget_left: int


@dataclass
class RepairTraceStep(Dumpable):
    round_idx: int
    action: str
    reason: str
    updated_modules: List[str]


@dataclass
class PipelineOutput(Dumpable):
    study_id: str
    predictions: List[dict]
    verifications: Dict[str, dict]
    retrievals: Dict[str, dict]
    conflicts: Dict[str, dict]
    coverage: float
    explanation_text: str
    audit_trace: List[dict]
    conformal_qhat_used: float
