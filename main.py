import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from src.config import DICT_MODELS, DICT_CAST_CATEGORY_COLS, TARGET_COLUMN
from src.data_loader import cargar_datos, preparar_datos, generar_csv_datos_preprocesados
from src.model_trainer import dividir_datos, construir_preprocesador, entrenar_modelos
from src.predictor import obtener_predicciones, obtener_predicciones_proba
from src.evaluator import (
    evaluar_modelos,
    generar_matriz_confusion,
    generar_grafica_importancia_variables,
    generar_mapa_calor_correlacion,
    generar_curva_aprendizaje
)

def entrenar_modelo_keras(X_train, y_train, X_test):
    """
    Entrena un modelo Keras transformando los datos primero con el preprocesador
    y aplicando EarlyStopping para evitar el sobreajuste.
    """
    preprocessor = construir_preprocesador(X_train, min_frequency=20)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Red neuronal secuencial con capas Dropout
    model_nn = models.Sequential([
        layers.Input(shape=(X_train_trans.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model_nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Guardamos el historial del entrenamiento
    history = model_nn.fit(
        X_train_trans,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
    )

    y_proba_nn = model_nn.predict(X_test_trans).flatten()
    y_pred_nn = (y_proba_nn > 0.5).astype(int)

    return y_pred_nn, y_proba_nn, history

def main():
    print("==================================================")
    print("         INICIANDO PIPELINE DE ML/DL              ")
    print("==================================================")

    # 1. Cargar y preparar datos
    print("\n[1/6] Cargando datos...")
    df_raw = cargar_datos()
    
    # Aplicar casting opcional si existen las columnas configuradas
    df_raw = df_raw.astype({k: v for k, v in DICT_CAST_CATEGORY_COLS.items() if k in df_raw.columns})

    print("[2/6] Limpiando datos...")
    df_clean = preparar_datos(df_raw)
    generar_csv_datos_preprocesados(df_clean)

    # Generar Mapa de Calor de Correlación
    generar_mapa_calor_correlacion(df_clean)

    # 2. Dividir dataset (Estratificado)
    print("[3/6] Dividiendo datos en Train / Test...")
    X_train, X_test, y_train, y_test = dividir_datos(df_clean, stratify=True)

    # 3. Entrenar modelos Scikit-Learn y LightGBM con GridSearchCV + Pipelines
    print("[4/6] Entrenando modelos tradicionales...")
    trained_pipelines = entrenar_modelos(DICT_MODELS, X_train, y_train)

    # 4. Generar predicciones usando el módulo predictor.py
    print("[5/6] Generando predicciones...")
    predictions = obtener_predicciones(trained_pipelines, X_test)
    predictions_proba = obtener_predicciones_proba(trained_pipelines, X_test)

    # Red Neuronal Keras (Opcional)
    try:
        print("\n--- Entrenando Red Neuronal Keras ---")
        y_pred_nn, y_proba_nn, history = entrenar_modelo_keras(X_train, y_train, X_test)
        predictions['Keras Neural Net'] = y_pred_nn
        predictions_proba['Keras Neural Net'] = y_proba_nn

        # Generar Curva de Aprendizaje para Keras
        generar_curva_aprendizaje(history)  

    except Exception as e:
        print(f"Omitiendo Keras por el siguiente motivo: {e}")

    # 5. Evaluación y Generación de Gráficos
    print("\n[6/6] Evaluando resultados y guardando gráficos...")
    df_resultados = evaluar_modelos(y_test, predictions, predictions_proba)

    print("\n================================================")
    print("                 RESULTADOS                       ")
    print("==================================================")
    print(df_resultados.to_string(index=False))
    print("==================================================")

    # Generar gráficos de salidas
    generar_matriz_confusion(y_test, predictions)

    if 'Random Forest' in trained_pipelines:
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        generar_grafica_importancia_variables(trained_pipelines['Random Forest'], cat_cols)

    print("\nPipeline completada con éxito.")

if __name__ == "__main__":
    main()