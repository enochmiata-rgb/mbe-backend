from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =========================================================
# BASE
# =========================================================

RiskLevel = Literal["low", "medium", "high"]
ReliabilityLevel = Literal["verified", "estimated", "simulated"]
GroundingRisk = Literal["low", "medium", "high"]


class Citation(BaseModel):
    label: str = ""
    title: str = ""
    sourceUrl: str = ""
    reason: str = ""
    sourceType: str = "external"
    hasRealText: bool = False
    confidence: float = 0.0


class GroundingSummary(BaseModel):
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    usedContext: bool = True
    usedSourcesWithRealTextFirst: bool = True
    confidence: float = 0.0
    limitations: List[str] = Field(default_factory=list)
    hallucinationRisk: GroundingRisk = "medium"


class EvidenceBundle(BaseModel):
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    grounding: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)


# =========================================================
# KPI
# =========================================================

class KpiSource(BaseModel):
    provider: str = ""
    sourceUrl: str = ""
    asOf: str = ""
    confidence: float = 0.0
    isLive: bool = False
    evidence: str = ""
    sourceMode: str = ""
    sourceCategory: str = ""


class KpiItem(BaseModel):
    key: str = ""
    title: str = ""
    value: Any = None
    unit: str = ""
    status: str = ""
    confidence: float = 0.0
    asOf: str = ""
    provider: str = ""
    sourceUrl: str = ""
    evidence: str = ""
    isLive: bool = False

    dataReliabilityLevel: str = "simulated"
    reliabilityScore: int = 0
    sourceSystem: str = ""
    sourceType: str = ""
    lastValidationAt: str = ""
    decisionImpact: str = ""
    decisionRecommendation: str = ""
    riskLevel: str = "medium"
    validationNotes: List[str] = Field(default_factory=list)

    source: Dict[str, Any] = Field(default_factory=dict)
    realism: Dict[str, Any] = Field(default_factory=dict)
    reliabilityEngine: Dict[str, Any] = Field(default_factory=dict)

    dataCollectionStatus: str = ""
    recommendedInternalSource: str = ""
    sourceGapEvidence: str = ""
    sourceStrategy: Dict[str, Any] = Field(default_factory=dict)


class KpiAlert(BaseModel):
    kpi: str = ""
    severity: str = ""
    message: str = ""
    riskLevel: str = "medium"
    decisionImpact: str = ""
    decisionRecommendation: str = ""
    evidence: List[str] = Field(default_factory=list)


class CrossKpiCheck(BaseModel):
    id: str = ""
    title: str = ""
    score: int = 0
    severity: str = ""
    message: str = ""
    recommendation: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    evaluatedAt: str = ""


class CrossKpiValidationResult(BaseModel):
    overallStatus: str = "warning"
    averageScore: float = 0.0
    criticalCount: int = 0
    warningCount: int = 0
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    topIssues: List[Dict[str, Any]] = Field(default_factory=list)
    generatedAt: str = ""


class KpisPayload(BaseModel):
    updatedAt: str = ""
    items: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    crossKpiValidation: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)


# =========================================================
# ASSISTANT
# =========================================================

class AssistantResponse(BaseModel):
    answer: str = ""
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    info: Optional[str] = None
    table: Optional[Dict[str, Any]] = None
    grounding: Dict[str, Any] = Field(default_factory=dict)
    crossKpiValidation: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)