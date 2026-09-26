import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from src.config import OUTPUTS_DIR, TARGET_COLUMN

def evaluar_modelos(y_test, y_pred: dict, y_pred_proba: dict):
    results = []

    for name, y_pred_model in y_pred.items():
        acc = accuracy_score(y_test, y_pred_model)
        prec = precision_score(y_test, y_pred_model, zero_division=0)
        rec = recall_score(y_test, y_pred_model, zero_division=0)
        f1 = f1_score(y_test, y_pred_model, zero_division=0)
        auc = None

        if name in y_pred_proba and y_pred_proba[name] is not None:
            proba = y_pred_proba[name]
            y_score = proba[:, 1] if proba.ndim > 1 else proba
            auc = roc_auc_score(y_test, y_score)

        results.append({
            'Modelo': name,
            'Accuracy': acc,
            'Precisión': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc
        })

    df_results = pd.DataFrame(results)
    return df_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

def generar_matriz_confusion(y_test, y_preds: dict):
    for model_name, y_pred_model in y_preds.items():
        cm = confusion_matrix(y_test, y_pred_model, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Cancelado', 'Cancelado'])
        disp.plot(cmap=plt.cm.Blues, values_format='.2%')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(OUTPUTS_DIR / f"matriz_confusion_{model_name}.png")
        plt.close()

def generar_grafica_importancia_variables(pipeline, categorical_columns: list):
    """
    Agrupa e imprime la importancia de variables extrayendo las características 
    directamente del ColumnTransformer dentro del Pipeline[cite: 1].
    """
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['prep']

    if not hasattr(model, 'feature_importances_'):
        print(f"El modelo {type(model).__name__} no soporta feature_importances_.")
        return

    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    df_imp = pd.DataFrame({'variable': feature_names, 'importancia': importances})

    # Función para mapear nombres transformados hacia la variable original[cite: 1]
    def agrupar_variable(nombre):
        partes = nombre.split('__', 1)
        resto = partes[1] if len(partes) > 1 else nombre
        for col in categorical_columns:
            if resto.startswith(col + '_'):
                return col
        return resto

    df_imp['variable_original'] = df_imp['variable'].apply(agrupar_variable)
    df_agr = df_imp.groupby('variable_original', as_index=False)['importancia'].sum().sort_values(by='importancia', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importancia', y='variable_original', data=df_agr.head(15), palette='viridis')
    plt.title('Top variables importantes (Agregadas)')
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.savefig(OUTPUTS_DIR / "importancia_variables.png")
    plt.close()


    # src/evaluator.py





def generar_mapa_calor_correlacion(df: pd.DataFrame):
    """
    Genera un mapa de calor con las principales correlaciones numéricas respecto a la variable objetivo.
    """
    plt.figure(figsize=(10, 8))
    # Seleccionamos solo columnas numéricas para evitar errores en corr()
    df_num = df.select_dtypes(include=['number'])
    
    if TARGET_COLUMN in df_num.columns:
        corr = df_num.corr()[[TARGET_COLUMN]].sort_values(by=TARGET_COLUMN, ascending=False)
        sns.heatmap(corr.head(15), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Top variables correlacionadas con la cancelación')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "top_variables_correlacionadas.png")
        plt.close()

def generar_curva_aprendizaje(history):
    """
    Guarda la gráfica de la curva de aprendizaje (pérdida/loss) del entrenamiento de Keras.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Loss train')
    plt.plot(history.history['val_loss'], label='Loss val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curva de aprendizaje - Keras')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "curva_aprendizaje.png")
    plt.close()