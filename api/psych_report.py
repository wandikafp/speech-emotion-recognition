"""
Phase 5 — Business Logic: Psychologist Pre-Assessment Report Generator.

Analyses emotion predictions over a full audio recording and produces
a structured report for psychologists.
"""

from typing import Optional
from collections import Counter


# ── Risk indicator rules ──────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "high_distress":   {"emotions": ["fearful", "sad", "disgust"],   "min_ratio": 0.50, "level": "HIGH"},
    "moderate_distress":{"emotions": ["fearful", "sad", "disgust"],  "min_ratio": 0.30, "level": "MODERATE"},
    "high_anger":      {"emotions": ["angry"],                        "min_ratio": 0.40, "level": "MODERATE"},
    "emotional_flat":  {"emotions": ["neutral", "calm"],              "min_ratio": 0.80, "level": "LOW"},
}

AROUSAL_VALENCE = {
    "angry":    {"arousal": "Tinggi",  "valence": "Negatif"},
    "fearful":  {"arousal": "Tinggi",  "valence": "Negatif"},
    "happy":    {"arousal": "Tinggi",  "valence": "Positif"},
    "surprised":{"arousal": "Tinggi",  "valence": "Positif"},
    "sad":      {"arousal": "Rendah",  "valence": "Negatif"},
    "disgust":  {"arousal": "Rendah",  "valence": "Negatif"},
    "calm":     {"arousal": "Rendah",  "valence": "Positif"},
    "neutral":  {"arousal": "Rendah",  "valence": "Positif"},
}

EMOTION_DESCRIPTIONS = {
    "neutral":  "Kondisi emosional yang stabil dan tidak menunjukkan ekspresi emosi yang menonjol.",
    "calm":     "Keadaan tenang yang mencerminkan stabilitas emosional dan regulasi diri yang baik.",
    "happy":    "Ekspresi kebahagiaan yang menunjukkan afek positif dan kesejahteraan emosional.",
    "sad":      "Ekspresi kesedihan yang dapat mengindikasikan depresi atau kehilangan.",
    "angry":    "Ekspresi kemarahan yang dapat mencerminkan frustrasi, konflik, atau agresi.",
    "fearful":  "Ekspresi ketakutan yang mungkin berkaitan dengan kecemasan, fobia, atau trauma.",
    "disgust":  "Ekspresi jijik yang dapat berkaitan dengan penolakan atau pengalaman negatif mendalam.",
    "surprised":"Ekspresi terkejut yang umumnya netral namun kontekstual.",
}

CLINICAL_NOTES = {
    "high_distress":    "⚠️ PERHATIAN KLINIS: Dominasi emosi negatif (fearful/sad/disgust) yang tinggi. Pertimbangkan asesmen depresi dan kecemasan lebih lanjut.",
    "moderate_distress":"⚡ Catatan: Proporsi emosi negatif moderat terdeteksi. Perlu eksplorasi lebih lanjut dalam sesi.",
    "high_anger":       "⚡ Catatan: Kemarahan dominan terdeteksi. Pertimbangkan asesmen regulasi emosi dan manajemen konflik.",
    "emotional_flat":   "ℹ️ Catatan: Afek yang sangat datar (neutral/calm mendominasi). Eksplorasi lebih lanjut untuk menyingkirkan alexithymia atau depresi terselubung.",
}


# ── Report generator ──────────────────────────────────────────────────────────

def generate_psych_report(predictions: list[dict],
                           client_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> dict:
    """
    Generate a psychologist pre-assessment report from a list of emotion predictions.

    Parameters
    ----------
    predictions : list[dict]
        Each dict is output from SERPredictor (dominant_emotion, confidence, emotion_scores).
    client_id : str, optional
    session_id : str, optional

    Returns
    -------
    dict — structured pre-assessment report
    """
    from datetime import datetime

    if not predictions:
        raise ValueError("predictions cannot be empty")

    n = len(predictions)
    all_emotions = [p["dominant_emotion"] for p in predictions]
    emotion_counts = Counter(all_emotions)

    # ── Emotion distribution ──────────────────────────────────────────────────
    emotion_distribution = {
        emo: {
            "count":   emotion_counts.get(emo, 0),
            "ratio":   round(emotion_counts.get(emo, 0) / n, 4),
            "percent": f"{emotion_counts.get(emo, 0) / n * 100:.1f}%",
        }
        for emo in AROUSAL_VALENCE.keys()
    }

    # Sort by frequency
    dominant_emotions = sorted(emotion_distribution.items(),
                                key=lambda x: x[1]["count"], reverse=True)

    # ── Dominant emotion ──────────────────────────────────────────────────────
    dominant_emotion = dominant_emotions[0][0]
    avg_confidence   = round(sum(p["confidence"] for p in predictions) / n, 4)

    # ── Arousal-Valence summary ───────────────────────────────────────────────
    arousal_counts  = Counter(AROUSAL_VALENCE[e]["arousal"] for e in all_emotions)
    valence_counts  = Counter(AROUSAL_VALENCE[e]["valence"] for e in all_emotions)
    dominant_arousal = arousal_counts.most_common(1)[0][0]
    dominant_valence = valence_counts.most_common(1)[0][0]

    # ── Emotion timeline (per segment) ───────────────────────────────────────
    timeline = []
    for i, pred in enumerate(predictions):
        timeline.append({
            "segment":          i + 1,
            "emotion":          pred["dominant_emotion"],
            "confidence":       round(pred["confidence"], 4),
            "arousal":          AROUSAL_VALENCE[pred["dominant_emotion"]]["arousal"],
            "valence":          AROUSAL_VALENCE[pred["dominant_emotion"]]["valence"],
        })

    # ── Emotion variability ────────────────────────────────────────────────────
    n_unique_emotions = len(emotion_counts)
    emotional_variability = "Tinggi" if n_unique_emotions >= 5 else \
                            "Sedang" if n_unique_emotions >= 3 else "Rendah"

    # ── Risk indicators ───────────────────────────────────────────────────────
    risk_flags = []
    clinical_notes = []
    for flag_name, rule in RISK_THRESHOLDS.items():
        ratio = sum(emotion_counts.get(e, 0) for e in rule["emotions"]) / n
        if ratio >= rule["min_ratio"]:
            risk_flags.append({
                "flag":    flag_name,
                "level":   rule["level"],
                "ratio":   round(ratio, 4),
                "emotions_involved": rule["emotions"],
            })
            clinical_notes.append(CLINICAL_NOTES.get(flag_name, ""))

    overall_risk = "HIGH" if any(f["level"] == "HIGH" for f in risk_flags) else \
                  "MODERATE" if any(f["level"] == "MODERATE" for f in risk_flags) else "LOW"

    # ── Assemble report ────────────────────────────────────────────────────────
    report = {
        "report_meta": {
            "generated_at":   datetime.now().isoformat(),
            "client_id":      client_id or "anonymous",
            "session_id":     session_id or "N/A",
            "total_segments": n,
            "avg_confidence": avg_confidence,
        },

        "dominant_emotion": {
            "emotion":      dominant_emotion,
            "description":  EMOTION_DESCRIPTIONS[dominant_emotion],
            "percent":      emotion_distribution[dominant_emotion]["percent"],
        },

        "arousal_valence": {
            "dominant_arousal": dominant_arousal,
            "dominant_valence": dominant_valence,
            "interpretation":   f"Secara keseluruhan, klien menunjukkan arousal {dominant_arousal.lower()} "
                                f"dengan valensi emosi yang cenderung {dominant_valence.lower()}.",
        },

        "emotion_distribution": {
            emo: data
            for emo, data in sorted(emotion_distribution.items(),
                                     key=lambda x: x[1]["count"], reverse=True)
            if data["count"] > 0
        },

        "emotional_variability": {
            "level":            emotional_variability,
            "n_unique_emotions": n_unique_emotions,
            "interpretation":   f"Variabilitas emosi {emotional_variability.lower()} — klien menunjukkan "
                                f"{n_unique_emotions} jenis emosi berbeda sepanjang rekaman.",
        },

        "risk_assessment": {
            "overall_risk_level": overall_risk,
            "risk_flags":         risk_flags,
            "clinical_notes":     [n for n in clinical_notes if n],
        },

        "timeline": timeline,

        "recommendations": _generate_recommendations(dominant_emotion, overall_risk, risk_flags),
    }

    return report


def _generate_recommendations(dominant_emotion: str,
                               risk_level: str,
                               risk_flags: list) -> list[str]:
    """Generate clinical recommendation notes based on findings."""
    recs = []

    base_recs = {
        "angry":   "Eksplorasi sumber kemarahan dan teknik regulasi emosi (anger management).",
        "fearful": "Asesmen kecemasan lebih lanjut (GAD-7, Beck Anxiety Inventory). Pertimbangkan pendekatan CBT.",
        "sad":     "Asesmen depresi (PHQ-9, BDI). Evaluasi sistem dukungan sosial klien.",
        "disgust": "Eksplorasi pengalaman traumatik atau penolakan. Pertimbangkan trauma-informed approach.",
        "calm":    "Kondisi emosional stabil. Fokus pada tujuan terapeutik jangka panjang.",
        "happy":   "Afek positif dominan. Verifikasi apakah mencerminkan keadaan klien yang sebenarnya.",
        "neutral": "Afek yang terkontrol. Eksplorasi apakah ini merupakan mekanisme pertahanan.",
        "surprised":"Kondisi bervariasi. Eksplorasi lebih lanjut konteks kehidupan klien.",
    }

    if dominant_emotion in base_recs:
        recs.append(base_recs[dominant_emotion])

    flag_names = [f["flag"] for f in risk_flags]
    if "high_distress" in flag_names:
        recs.append("Prioritaskan sesi awal untuk membangun rapport dan rasa aman.")
        recs.append("Pertimbangkan rujukan ke psikiater jika diperlukan intervensi farmakologis.")
    if "emotional_flat" in flag_names:
        recs.append("Eksplorasi lebih dalam kemungkinan alexithymia atau disosiasi emosional.")
    if "high_anger" in flag_names:
        recs.append("Kaji riwayat agresivitas dan dampaknya pada hubungan interpersonal.")

    recs.append("Laporan ini bersifat indikatif. Interpretasi klinis harus mempertimbangkan konteks lengkap klien.")
    return recs


def format_report_text(report: dict) -> str:
    """Format report as human-readable text for display."""
    meta  = report["report_meta"]
    dom   = report["dominant_emotion"]
    av    = report["arousal_valence"]
    risk  = report["risk_assessment"]

    lines = [
        "=" * 60,
        "  LAPORAN PRE-ASSESSMENT PSIKOLOG",
        "  Speech Emotion Recognition System",
        "=" * 60,
        f"  Tanggal    : {meta['generated_at'][:19].replace('T', ' ')}",
        f"  Client ID  : {meta['client_id']}",
        f"  Session ID : {meta['session_id']}",
        f"  Segments   : {meta['total_segments']} | Avg Confidence: {meta['avg_confidence']:.2f}",
        "",
        "── EMOSI DOMINAN ─────────────────────────────────────────",
        f"  {dom['emotion'].upper()} ({dom['percent']})",
        f"  {dom['description']}",
        "",
        "── AROUSAL-VALENCE ───────────────────────────────────────",
        f"  Arousal  : {av['dominant_arousal']}",
        f"  Valence  : {av['dominant_valence']}",
        f"  {av['interpretation']}",
        "",
        "── DISTRIBUSI EMOSI ──────────────────────────────────────",
    ]
    for emo, data in report["emotion_distribution"].items():
        bar = "█" * int(data["ratio"] * 30)
        lines.append(f"  {emo:12s}: {data['percent']:6s} {bar}")

    lines += [
        "",
        "── RISK ASSESSMENT ───────────────────────────────────────",
        f"  Overall Risk Level: {risk['overall_risk_level']}",
    ]
    for flag in risk["risk_flags"]:
        lines.append(f"  [{flag['level']}] {flag['flag']} ({flag['ratio']*100:.0f}%)")
    for note in risk["clinical_notes"]:
        lines.append(f"  {note}")

    lines += ["", "── REKOMENDASI ───────────────────────────────────────────"]
    for i, rec in enumerate(report["recommendations"], 1):
        lines.append(f"  {i}. {rec}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
