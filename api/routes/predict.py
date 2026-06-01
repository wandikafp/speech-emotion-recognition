import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
import base64

from api.dependencies import get_predictor
from api.router import route_contact_center, get_dominant_routing
from api.psych_report import generate_psych_report
from api.schema import (
    AudioPredictRequest, MultiSegmentRequest, UseCase
)

router = APIRouter(tags=["Prediction"])

def _predict_segment(predictor, audio_b64: str, audio_format: str = "wav") -> tuple[dict, float]:
    """Run prediction and return (result, latency_ms)."""
    t0 = time.perf_counter()
    result = predictor.predict_from_base64(audio_b64, audio_format)
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms

@router.post("/api/v1/predict")
async def predict(request: AudioPredictRequest, predictor=Depends(get_predictor)):
    """
    Predict emotion from a single audio segment.
    Returns emotion scores + optional routing / psych report based on use_case.
    """
    result, latency_ms = _predict_segment(predictor, request.audio_base64, request.audio_format)

    response = {
        **result,
        "processing_time_ms": round(latency_ms, 2),
        "routing": None,
        "psych_report": None,
    }

    if request.use_case == UseCase.contact_center:
        response["routing"] = route_contact_center(result)

    elif request.use_case == UseCase.psych_assessment:
        report = generate_psych_report(
            [result],
            client_id=request.client_id,
            session_id=request.session_id,
        )
        response["psych_report"] = report

    return JSONResponse(content=response)

@router.post("/api/v1/predict/session")
async def predict_session(request: MultiSegmentRequest, predictor=Depends(get_predictor)):
    """
    Analyse multiple audio segments (e.g. a full call recording split into chunks).
    Returns per-segment predictions + aggregate analysis.
    """
    if not request.segments:
        raise HTTPException(status_code=400, detail="segments cannot be empty")

    t0 = time.perf_counter()
    segment_results = []
    for seg_b64 in request.segments:
        result, _ = _predict_segment(predictor, seg_b64, request.audio_format)
        segment_results.append(result)
    total_latency = (time.perf_counter() - t0) * 1000

    response = {
        "segments":           segment_results,
        "processing_time_ms": round(total_latency, 2),
        "dominant_routing":   None,
        "psych_report":       None,
    }

    if request.use_case == UseCase.contact_center:
        routed = [route_contact_center(r) for r in segment_results]
        response["dominant_routing"] = get_dominant_routing(routed)

    elif request.use_case == UseCase.psych_assessment:
        response["psych_report"] = generate_psych_report(
            segment_results,
            client_id=request.client_id,
            session_id=request.session_id,
        )

    return JSONResponse(content=response)

@router.post("/api/v1/predict/file")
async def predict_file(
    file: UploadFile = File(..., description="Upload a .wav audio file"),
    use_case: UseCase = Query(UseCase.general),
    predictor=Depends(get_predictor)
):
    """
    Predict emotion directly from an uploaded audio file (multipart/form-data).
    """
    audio_bytes = await file.read()
    audio_b64   = base64.b64encode(audio_bytes).decode("utf-8")
    ext         = file.filename.rsplit(".", 1)[-1].lower() if file.filename else "wav"

    result, latency_ms = _predict_segment(predictor, audio_b64, ext)

    response = {
        **result,
        "filename":           file.filename,
        "processing_time_ms": round(latency_ms, 2),
        "routing": None,
    }

    if use_case == UseCase.contact_center:
        response["routing"] = route_contact_center(result)

    return JSONResponse(content=response)
