from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from api.dependencies import get_predictor
from api.schema import AudioPredictRequest
from api.router import route_contact_center
from api.routes.predict import _predict_segment

router = APIRouter(tags=["Contact Center"])

@router.post("/api/v1/contact-center/route")
async def contact_center_route(request: AudioPredictRequest, predictor=Depends(get_predictor)):
    """
    Dedicated endpoint for contact center routing.
    Predicts emotion and returns agent routing recommendation.
    """
    result, latency_ms = _predict_segment(predictor, request.audio_base64, request.audio_format)
    routing = route_contact_center(result)
    return JSONResponse(content={
        **routing,
        "processing_time_ms": round(latency_ms, 2),
    })
