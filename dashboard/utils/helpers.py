import base64

def audio_to_b64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")

def predict_from_bytes(predictor, audio_bytes: bytes, fmt="wav") -> dict:
    b64 = audio_to_b64(audio_bytes)
    return predictor.predict_from_base64(b64, fmt)
