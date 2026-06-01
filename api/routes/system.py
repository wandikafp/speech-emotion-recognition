from fastapi import APIRouter, Depends
from api.dependencies import get_predictor, _find_latest_model, predictor_state
from api.schema import HealthResponse, ModelInfo

router = APIRouter(tags=["System"])

@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check — returns model status."""
    model_path = _find_latest_model() or "N/A"
    return HealthResponse(
        status="ok" if predictor_state.predictor is not None else "degraded",
        model_loaded=predictor_state.predictor is not None,
        model_path=model_path,
    )

@router.get("/api/v1/model/info", response_model=ModelInfo)
async def model_info(predictor=Depends(get_predictor)):
    """Return model architecture metadata."""
    input_shape = list(predictor.model.input_shape[1:])
    return ModelInfo(
        model_name=predictor.model.name,
        model_path=_find_latest_model() or "N/A",
        n_classes=len(predictor.classes),
        classes=predictor.classes,
        input_shape=input_shape,
    )
