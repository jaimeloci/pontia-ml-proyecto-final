from src.config import OUTPUTS_DIR
from src.predictor import predecir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score
)
# from tensorflow.keras import layers, models

def evaluar_modelos(models, X_test, y_test):

    results = []

    for name, model in models.items():
        y_pred = predecir(model, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Modelo': name, 'Precisión': accuracy})

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Precisión', ascending=False).reset_index(drop=True)
    return df_results

def distribucion_variable_objetivo(df, target_column):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title(f'Distribución de la variable objetivo ({target_column})')
    plt.xticks([0, 1], ['No Cancelado', 'Cancelado'])
    plt.ylabel('Cantidad de reservas')
    plt.savefig(OUTPUTS_DIR / f"distribucion_{target_column}.png")
    plt.close()

# Calculamos las métricas de evaluación
def calcular_metricas_evaluacion(y_prediccion: np.ndarray, y_real: np.ndarray, verbose: bool = True):
    """Calcula las métricas de evaluación para un modelo de regresión.
    
    Calcula cuatro métricas comunes para evaluar modelos de regresión: MSE (Error Cuadrático Medio),
    RMSE (Raíz del Error Cuadrático Medio), MAE (Error Absoluto Medio) y R² (Coeficiente de determinación).
    Opcionalmente imprime los resultados en un formato legible.
    
    Args:
        y_prediccion (np.ndarray): Valores predichos por el modelo.
        y_real (np.ndarray): Valores reales observados.
        verbose (bool, optional): Si es True, imprime las métricas calculadas. Por defecto es True.
    
    Returns:
        tuple[float, float, float, float]: Una tupla con cuatro valores en el siguiente orden:
            - mse: Error cuadrático medio.
            - rmse: Raíz del error cuadrático medio.
            - mae: Error absoluto medio.
            - r2: Coeficiente de determinación.
    
    Example:
        >>> mse, rmse, mae, r2 = calcular_metricas_evaluacion(modelo.predict(X_test), y_test)
        >>> print(f"R²: {r2:.4f}")
    """
    
    mse = mean_squared_error(y_real, y_prediccion)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_prediccion)
    r2 = r2_score(y_real, y_prediccion)

    if verbose:
        print("\nEvaluación del modelo:")
        print(f"MSE (Error cuadrático medio): {mse:.4f}")
        print(f"RMSE (Raíz del error cuadrático medio): {rmse:.4f}")
        print(f"MAE (Error absoluto medio): {mae:.4f}")
        print(f"R² (Coeficiente de determinación): {r2:.4f}")
        print(f"El modelo explica aproximadamente el {r2:.2%} de la varianza")
    
    return mse, rmse, mae, r2