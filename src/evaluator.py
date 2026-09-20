from src.config import OUTPUTS_DIR
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, 
    precision_score, recall_score, f1_score,  
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)

def evaluar_modelos(y_test, y_pred, y_pred_proba):

    results = []

    for name, y_pred_model in y_pred.items():
        acc = accuracy_score(y_test, y_pred_model)
        prec = precision_score(y_test, y_pred_model)
        rec = recall_score(y_test, y_pred_model)
        f1 = f1_score(y_test, y_pred_model)
        auc = None
        mse, rmse, mae, r2 = calcular_metricas_evaluacion(y_pred_model, y_test, False)
        results.append({'Modelo': name,
                        'Accuracy': acc, 'Precisión': prec, 'Recall': rec, 'F1': f1,
                        'AUC': auc, 
                        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

    for name, y_pred_model_proba in y_pred_proba.items():
        auc = roc_auc_score(y_test, y_pred_model_proba[:, 1])
        for result in results:
            if result['Modelo'] == name:
                result['AUC'] = auc
                break

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    return df_results

# Metodo para generar fichero png con una visualización de la distribucion la variable objetivo
def distribucion_variable_objetivo(df, target_column):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title(f'Distribución de la variable objetivo ({target_column})')
    plt.xticks([0, 1], ['No Cancelado', 'Cancelado'])
    plt.ylabel('Cantidad de reservas')
    plt.savefig(OUTPUTS_DIR / f"distribucion_{target_column}.png")
    plt.close()

def generar_matriz_confusion(y_test, y_pred):
    for model_name, y_pred_model in y_pred.items():
        cm = confusion_matrix(y_test, y_pred_model)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Cancelado', 'Cancelado'])
        disp.plot(cmap=plt.cm.Blues,values_format='.2%')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(OUTPUTS_DIR / f"matriz_confusion_{model_name}.png")
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