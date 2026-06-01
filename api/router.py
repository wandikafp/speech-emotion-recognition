"""
Phase 5 — Business Logic: Contact Center Agent Routing.

Maps predicted emotion + confidence → routing recommendation.
"""

from typing import Optional


# ── Emotion → Agent type mapping ─────────────────────────────────────────────

ROUTING_RULES = [
    # (emotion, min_confidence, agent_type, priority, description)
    ("angry",    0.60, "de_escalation",  1, "Senior / De-escalation Specialist"),
    ("fearful",  0.60, "empathetic",     2, "Empathetic Support Specialist"),
    ("sad",      0.60, "empathetic",     2, "Empathetic Support Specialist"),
    ("disgust",  0.60, "quality",        3, "Quality Assurance Agent"),
    ("happy",    0.50, "standard",       4, "Standard Agent"),
    ("calm",     0.50, "standard",       4, "Standard Agent"),
    ("neutral",  0.50, "general",        5, "General Queue"),
    ("surprised",0.50, "general",        5, "General Queue"),
]

# Confidence threshold below which routing goes to manual review
LOW_CONFIDENCE_THRESHOLD = 0.40

# Human-readable descriptions per agent type
AGENT_DESCRIPTIONS = {
    "de_escalation": "Agent Senior / De-escalation Specialist — Terlatih menangani pelanggan marah dan situasi konflik tinggi.",
    "empathetic":    "Empathetic Support Specialist — Terlatih memberikan dukungan emosional untuk pelanggan yang takut atau sedih.",
    "quality":       "Quality Assurance Agent — Menangani keluhan kompleks dan memastikan resolusi memuaskan.",
    "standard":      "Standard Agent — Menangani kebutuhan umum pelanggan yang dalam kondisi positif.",
    "general":       "General Queue — Antrean umum untuk kondisi netral atau tidak tentu.",
    "manual_review": "Priority Queue / Manual Review — Tingkat kepercayaan prediksi rendah, perlu evaluasi manual.",
}

# Arousal-Valence mapping untuk konteks psikologi
AROUSAL_VALENCE = {
    "angry":    {"arousal": "high",  "valence": "negative"},
    "fearful":  {"arousal": "high",  "valence": "negative"},
    "happy":    {"arousal": "high",  "valence": "positive"},
    "surprised":{"arousal": "high",  "valence": "positive"},
    "sad":      {"arousal": "low",   "valence": "negative"},
    "disgust":  {"arousal": "low",   "valence": "negative"},
    "calm":     {"arousal": "low",   "valence": "positive"},
    "neutral":  {"arousal": "low",   "valence": "positive"},
}


# ── Main routing function ─────────────────────────────────────────────────────

def route_contact_center(prediction: dict) -> dict:
    """
    Apply contact center routing logic to a SER prediction.

    Parameters
    ----------
    prediction : dict
        Output from SERPredictor (dominant_emotion, confidence, emotion_scores).

    Returns
    -------
    dict with routing fields added:
        - agent_type          : str  (de_escalation / empathetic / standard / general / manual_review)
        - agent_description   : str
        - routing_priority    : int  (1=highest)
        - routing_reason      : str
        - requires_escalation : bool
        - arousal             : str
        - valence             : str
    """
    emotion    = prediction["dominant_emotion"]
    confidence = prediction["confidence"]

    # Low-confidence → manual review
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return {
            **prediction,
            "agent_type":          "manual_review",
            "agent_description":   AGENT_DESCRIPTIONS["manual_review"],
            "routing_priority":    0,
            "routing_reason":      f"Tingkat kepercayaan terlalu rendah ({confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD}). Routing manual diperlukan.",
            "requires_escalation": True,
            **AROUSAL_VALENCE.get(emotion, {"arousal": "unknown", "valence": "unknown"}),
        }

    # Match routing rule
    for emo, min_conf, agent_type, priority, description in ROUTING_RULES:
        if emotion == emo and confidence >= min_conf:
            requires_escalation = agent_type in ("de_escalation", "empathetic")
            return {
                **prediction,
                "agent_type":          agent_type,
                "agent_description":   AGENT_DESCRIPTIONS[agent_type],
                "routing_priority":    priority,
                "routing_reason":      f"Emosi '{emotion}' terdeteksi dengan confidence {confidence:.2f} ≥ {min_conf}. Diarahkan ke {description}.",
                "requires_escalation": requires_escalation,
                **AROUSAL_VALENCE.get(emotion, {"arousal": "unknown", "valence": "unknown"}),
            }

    # Fallback if confidence below rule's min_conf → general queue
    return {
        **prediction,
        "agent_type":          "general",
        "agent_description":   AGENT_DESCRIPTIONS["general"],
        "routing_priority":    5,
        "routing_reason":      f"Emosi '{emotion}' dengan confidence {confidence:.2f} tidak memenuhi threshold. Diarahkan ke General Queue.",
        "requires_escalation": False,
        **AROUSAL_VALENCE.get(emotion, {"arousal": "unknown", "valence": "unknown"}),
    }


def batch_route(predictions: list[dict]) -> list[dict]:
    """Route a list of predictions (e.g., for a full call session)."""
    return [route_contact_center(p) for p in predictions]


def get_dominant_routing(routed_segments: list[dict]) -> dict:
    """
    Given multiple routed segments from a call, determine the dominant routing
    by selecting the highest-priority (lowest priority number) result.
    """
    if not routed_segments:
        raise ValueError("routed_segments cannot be empty")

    # Sort by routing_priority ascending (lower = higher priority)
    sorted_segments = sorted(routed_segments, key=lambda x: x.get("routing_priority", 99))
    dominant = sorted_segments[0]

    # Emotion frequency for summary
    from collections import Counter
    emotion_counts = Counter(s["dominant_emotion"] for s in routed_segments)

    return {
        **dominant,
        "session_summary": {
            "total_segments":  len(routed_segments),
            "emotion_counts":  dict(emotion_counts),
            "dominant_emotion_in_session": emotion_counts.most_common(1)[0][0],
        }
    }
