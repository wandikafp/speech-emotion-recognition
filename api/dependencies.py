import os
import glob
from typing import Optional
from fastapi import HTTPException
from src.predict import SERPredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PredictorState:
    predictor: Optional[SERPredictor] = None

predictor_state = PredictorState()

def _find_latest_model() -> Optional[str]:
    """Find the most recently saved .keras model file in models/."""
    models_dir = os.path.join(BASE_DIR, "models")
    keras_files = glob.glob(os.path.join(models_dir, "*.keras"))
    if not keras_files:
        return None
    return max(keras_files, key=os.path.getmtime)

def load_predictor():
    """Load model at startup."""
    model_path = os.getenv("SER_MODEL_PATH") or _find_latest_model()
    if model_path and os.path.exists(model_path):
        try:
            predictor_state.predictor = SERPredictor(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
    else:
        print("⚠️  No model found. Set SER_MODEL_PATH or place a .keras file in models/")

def unload_predictor():
    """Clean up on shutdown."""
    predictor_state.predictor = None
    print("Model unloaded.")

def get_predictor() -> SERPredictor:
    """Dependency to get the loaded predictor."""
    if predictor_state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    return predictor_state.predictor
