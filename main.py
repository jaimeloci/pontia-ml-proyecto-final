

from src.config import DICT_MODELS, IRRELEVANT_COLUMNS, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
from src.data_loader import preparar_datos,generar_csv_datos_preprocesados, cargar_datos
from src.model_trainer import dividir_datos, entrenar_modelos, escalar_datos
from src.evaluator import distribucion_variable_objetivo,evaluar_modelos, generar_matriz_confusion
from src.predictor import obtener_predicciones, obtener_predicciones_proba

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
   
    # Exportar matriz de confusión para el mejor modelo
    generar_matriz_confusion(y_test, predictions)

    # Guardar resultados del mejor modelo
    #save_confusion_matrix(models[best_model_name], X_test, y_test, best_model_name)
    #save_best_model(models[best_model_name], best_model_name)


if __name__ == "__main__":
    main()