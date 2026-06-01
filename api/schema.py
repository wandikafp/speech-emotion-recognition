"""
Pydantic schemas for the SER FastAPI.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class UseCase(str, Enum):
    general         = "general"
    contact_center  = "contact_center"
    psych_assessment = "psych_assessment"


# ── Request schemas ───────────────────────────────────────────────────────────

class AudioPredictRequest(BaseModel):
    """Single-segment prediction request."""
    audio_base64:  str       = Field(..., description="Base64-encoded WAV audio")
    audio_format:  str       = Field("wav", description="Audio format (wav, mp3, etc.)")
    use_case:      UseCase   = Field(UseCase.general, description="Intended use case")
    client_id:     Optional[str] = Field(None, description="Client identifier (optional)")
    session_id:    Optional[str] = Field(None, description="Session identifier (optional)")


class MultiSegmentRequest(BaseModel):
    """Multi-segment prediction request (for full call / long recording)."""
    segments:      List[str] = Field(..., description="List of base64-encoded audio segments")
    audio_format:  str       = Field("wav")
    use_case:      UseCase   = Field(UseCase.contact_center)
    client_id:     Optional[str] = None
    session_id:    Optional[str] = None


# ── Response schemas ──────────────────────────────────────────────────────────

class EmotionScores(BaseModel):
    neutral:   float
    calm:      float
    happy:     float
    sad:       float
    angry:     float
    fearful:   float
    disgust:   float
    surprised: float


class RoutingInfo(BaseModel):
    agent_type:          str
    agent_description:   str
    routing_priority:    int
    routing_reason:      str
    requires_escalation: bool
    arousal:             str
    valence:             str


class RiskFlag(BaseModel):
    flag:              str
    level:             str
    ratio:             float
    emotions_involved: List[str]


class PsychReport(BaseModel):
    dominant_emotion:       Dict
    arousal_valence:        Dict
    emotion_distribution:   Dict
    emotional_variability:  Dict
    risk_assessment:        Dict
    recommendations:        List[str]
    timeline:               List[Dict]


class EmotionPrediction(BaseModel):
    """Response for single-segment prediction."""
    dominant_emotion:    str
    confidence:          float
    emotion_scores:      Dict[str, float]
    processing_time_ms:  float
    # Optional fields populated based on use_case
    routing:             Optional[RoutingInfo]    = None
    psych_report:        Optional[PsychReport]   = None


class SessionAnalysis(BaseModel):
    """Response for multi-segment analysis."""
    segments:            List[EmotionPrediction]
    dominant_routing:    Optional[RoutingInfo]   = None
    psych_report:        Optional[PsychReport]   = None
    processing_time_ms:  float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_path:    str
    version:       str = "1.0.0"


class ModelInfo(BaseModel):
    model_name:    str
    model_path:    str
    n_classes:     int
    classes:       List[str]
    input_shape:   List[int]
    version:       str = "1.0.0"
