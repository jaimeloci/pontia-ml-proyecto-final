

from src.config import DICT_MODELS, IRRELEVANT_COLUMNS, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
from src.data_loader import preparar_datos,generar_csv_datos_preprocesados, cargar_datos
from src.model_trainer import dividir_datos, entrenar_modelos, escalar_datos
from src.evaluator import distribucion_variable_objetivo,evaluar_modelos
#from src.predictor import save_confusion_matrix, save_best_model

def main():
    print("======== PIPELINE INICIADA ==========")
    
    print(f"\n[1/8] Cargando datos brutos desde el archivo: {RAW_DATA_PATH}...") 
    df_raw = cargar_datos()

    print("\nGenerando imagen de distribución de la variable objetivo...")
    distribucion_variable_objetivo(df_raw.drop(columns=IRRELEVANT_COLUMNS), TARGET_COLUMN)

    print("\n[2/8] Preprocesando datos...")
    df_preprocessed = preparar_datos(df_raw)

    print(f"\n[3/8] Guardando datos preprocesados en el archivo: {PROCESSED_DATA_PATH}...")
    generar_csv_datos_preprocesados(df_preprocessed)

    print("\n[4/8] Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = dividir_datos(df_preprocessed)

    print("\n[5/8] Escalando datos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = escalar_datos(X_train, X_test, y_train, y_test)

    print("\n[6/8] Inicializando modelos...")
    models = DICT_MODELS

    # Entrenar modelos usando GridSearchCV y obtener los mejores hiperparámetros
    print("\n[7/8] Entrenando modelos...")
    trained_models = entrenar_modelos(models, X_train, y_train)
    
    # Evaluar modelos
    print("\n[8/8] Evaluando modelos...")
    df_results = evaluar_modelos(trained_models, X_test, y_test)
    
    print("\n=== RESULTADOS ===")
    print(df_results.to_string(index=False))
    
    # # Exportar matriz de confusión y guardar el mejor modelo
    # best_model_name = df_results.iloc[0]['Modelo']
    # print(f"\n[8/8] Exportando resultados para el mejor modelo: {best_model_name}")

    # Guardar resultados del mejor modelo
    #save_confusion_matrix(models[best_model_name], X_test, y_test, best_model_name)
    #save_best_model(models[best_model_name], best_model_name)


if __name__ == "__main__":
    main()