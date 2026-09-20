from src.config import DICT_MODELS, IRRELEVANT_COLUMNS, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
from src.data_loader import cargar_datos, preparar_datos, generar_csv_datos_preprocesados 
from src.model_trainer import dividir_datos, escalar_datos, entrenar_modelos
from src.predictor import obtener_predicciones, obtener_predicciones_proba
from src.evaluator import distribucion_variable_objetivo,evaluar_modelos, generar_matriz_confusion

def main():
    print("======== PIPELINE INICIADA ==========")
 
    print(f"\n[1/9] Cargando datos brutos desde el archivo: {RAW_DATA_PATH}...") 
    df_raw = cargar_datos()

    print("\nGenerando imagen de distribución de la variable objetivo...")
    distribucion_variable_objetivo(df_raw.drop(columns=IRRELEVANT_COLUMNS), TARGET_COLUMN)

    print("\n[2/9] Preprocesando datos...")
    df_preprocessed = preparar_datos(df_raw)

    print(f"\n[3/9] Guardando datos preprocesados en el archivo: {PROCESSED_DATA_PATH}...")
    generar_csv_datos_preprocesados(df_preprocessed)

    print("\n[4/9] Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = dividir_datos(df_preprocessed)

    print("\n[5/9] Escalando datos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = escalar_datos(X_train, X_test, y_train, y_test)

    print("\n[6/9] Inicializando modelos...")
    models = DICT_MODELS

    print("\n[7/9] Entrenando modelos...")
    trained_models = entrenar_modelos(models, X_train, y_train)

    print("\n[8/9] Obteniendo predicciones de los modelos...")
    predictions = obtener_predicciones(trained_models, X_test)
    predictions_proba = obtener_predicciones_proba(trained_models, X_test)
    
    print("\n[9/9] Evaluando modelos...")
    df_results = evaluar_modelos(y_test, predictions, predictions_proba)
    
    print("\n=== RESULTADOS ===")
    print(df_results.to_string(index=False))
    print("=" * 100)
    for i, row in df_results.iterrows():
        print(f"Modelo: {row['Modelo']}")
        print(f"Accuracy: {row['Accuracy']:.4f}")
        print(f"Precisión: {row['Precisión']:.4f}")
        print(f"Recall: {row['Recall']:.4f}")
        print(f"F1: {row['F1']:.4f}")
        print(f"AUC: {row['AUC']:.4f}")
        print(f"MSE (Error cuadrático medio): {row['MSE']:.4f}")
        print(f"RMSE (Raíz del error cuadrático medio): {row['RMSE']:.4f}")
        print(f"MAE (Error absoluto medio): {row['MAE']:.4f}")
        print(f"R2 (Coeficiente de determinación): {row['R2']:.4f}")
        print(f"El modelo explica aproximadamente el {row['R2']:.2%} de la varianza")
        print("-" * 60)
   
    # Exportar matriz de confusión para el mejor modelo
    generar_matriz_confusion(y_test, predictions)


    print("\n=== RESULTADOS TENSORFLOW (Keras) ===")
    keras_model(X_train, y_train, X_test, y_test)

    # Guardar resultados del mejor modelo
    #save_confusion_matrix(models[best_model_name], X_test, y_test, best_model_name)
    #save_best_model(models[best_model_name], best_model_name)


from tensorflow.keras import layers, models
import tensorflow as tf
def keras_model(X_train, y_train, X_test, y_test):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)), name='input_layer')
    model.add(layers.Dense(128, activation='relu'), name='hidden_layer_1')
    model.add(layers.Dense(64, activation='relu'), name='hidden_layer_2')
    model.add(layers.Dense(32, activation='relu'), name='hidden_layer_3')
    model.add(layers.Dense(1, activation='sigmoid'), name='output_layer')
    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer_adam, loss='binary_crossentropy', metrics=['accuracy'])

    history_dp = model.fit(X_train, y_train, epochs=150, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test)

    # Evaluamos el modelo en el conjunto de prueba
    loss_dp, accuracy_dp = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss_dp:.4f}, Test Accuracy: {accuracy_dp:.4f}")
    return model

if __name__ == "__main__":
    main()