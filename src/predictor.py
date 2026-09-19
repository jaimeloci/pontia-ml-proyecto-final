
def obtener_predicciones(models, X):
    predictions = {}
    for model_name in models:
        model = models[model_name]

        predictions[model_name] = model.predict(X)
    return predictions

def obtener_predicciones_proba(models, X):
    predictions_proba = {}
    for model_name in models:
        model = models[model_name]
        
        predictions_proba[model_name] = model.predict_proba(X)
    return predictions_proba