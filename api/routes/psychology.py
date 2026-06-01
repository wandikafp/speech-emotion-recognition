import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from api.dependencies import get_predictor
from api.schema import MultiSegmentRequest
from api.psych_report import generate_psych_report
from api.routes.predict import _predict_segment

router = APIRouter(tags=["Psychology"])

@router.post("/api/v1/psych/assessment")
async def psych_assessment(request: MultiSegmentRequest, predictor=Depends(get_predictor)):
    """
    Dedicated endpoint for psychologist pre-assessment.
    Accepts multiple audio segments and returns a structured clinical report.
    """
    if not request.segments:
        raise HTTPException(status_code=400, detail="At least one audio segment is required")

    t0 = time.perf_counter()
    preds = []
    for seg_b64 in request.segments:
        result, _ = _predict_segment(predictor, seg_b64, request.audio_format)
        preds.append(result)
    total_latency = (time.perf_counter() - t0) * 1000

    report = generate_psych_report(
        preds,
        client_id=request.client_id,
        session_id=request.session_id,
    )

    return JSONResponse(content={
        **report,
        "processing_time_ms": round(total_latency, 2),
    })
