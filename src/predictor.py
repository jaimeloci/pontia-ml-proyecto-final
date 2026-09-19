
import joblib
from src.config import MODELS_DIR

def save_best_model(model, model_name: str):
    """Guarda el objeto del modelo entrenado en la carpeta models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.joblib"
    
    if 'Keras' not in model_name:
        joblib.dump(model, filepath)
        print(f"Modelo guardado exitosamente en: {filepath}")

def load_trained_model(model_name: str):
    """Carga un modelo entrenado."""
    filepath = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.joblib"
    return joblib.load(filepath)