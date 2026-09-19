

from src.data_loader import preparar_datos,generar_csv_datos_preprocesados, cargar_datos
from src.model_trainer import dividir_datos, escalar_datos
from src.config import IRRELEVANT_COLUMNS, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
from src.evaluator import distribucion_variable_objetivo
#from src.predictor import save_confusion_matrix, save_best_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
    X_train, X_test, y_train, y_test, preprocessor = escalar_datos(X_train, X_test, y_train, y_test)

    print("\n[6/8] Inicializando modelos...")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    
    # # Entrenar y evaluar
    # print("\n[7/8] Entrenando y evaluando modelos...")
    # df_results = evaluar_modelo(models, X_train, y_train, X_test, y_test)
    
    # print("\n=== RESULTADOS ===")
    # print(df_results.to_string(index=False))
    
    # # Exportar matriz de confusión y guardar el mejor modelo
    # best_model_name = df_results.iloc[0]['Modelo']
    # print(f"\n[8/8] Exportando resultados para el mejor modelo: {best_model_name}")

    # Guardar resultados del mejor modelo
    #save_confusion_matrix(models[best_model_name], X_test, y_test, best_model_name)
    #save_best_model(models[best_model_name], best_model_name)


if __name__ == "__main__":
    main()