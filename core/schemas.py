from typing import List, Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    canonical_claim_id: str
    claim_text: str
    first_seen_ts: str
    last_seen_ts: str
    mention_count: int
    source_types: List[str]
    support_count: int
    contradict_count: int
    confidence: float
    status: str
    linked_evidence_ids: List[str] = Field(default_factory=list)
    linked_media_ids: List[str] = Field(default_factory=list)
    trend_score: float = 0.0
    contradiction_ratio: float = 0.0
    meme_variant_count: int = 0
    volatility_score: float = 0.0
    alert_level: str = "low"
    last_agent_update_ts: Optional[str] = None
    language: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)


class EvidenceSnippet(BaseModel):
    evidence_id: str
    claim_id: str
    snippet_text: str
    stance: str
    source_id: str
    source_type: str
    timestamp: str
    url: Optional[str] = None
    credibility_tier: str = "C"


class MemeMedia(BaseModel):
    media_id: str
    source_id: str
    timestamp: str
    phash: str
    ocr_text: str
    linked_claim_ids: List[str]
    template_cluster_id: Optional[str] = None


class RetrievalTrace(BaseModel):
    collection: str
    point_id: str
    score: float
    path: str
    filters: Optional[str] = None
    payload_preview: Optional[str] = None


class ResponseBundle(BaseModel):
    query: str
    claims: List[dict]
    evidence: dict
    similar_memes: List[dict]
    timeline: List[dict]
    trace: List[RetrievalTrace]
