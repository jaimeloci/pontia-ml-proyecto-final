from src.predictor import predecir
from src.model_trainer import entrenar_modelo
from sklearn.metrics import accuracy_score
import pandas as pd

def evaluar_modelo(models, X_train, y_train, X_test, y_test):

    results = []

    for name, model in models.items():
        model = entrenar_modelo(model, X_train, y_train)
        y_pred = predecir(model, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Modelo': name, 'Precisión': accuracy})

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Precisión', ascending=False).reset_index(drop=True)
    return df_results