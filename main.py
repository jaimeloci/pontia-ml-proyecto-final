

from src.data_loader import preparar_datos, dividir_datos, escalar_datos, generar_datos_procesados, cargar_datos
from src.evaluator import evaluar_modelo
#from src.predictor import save_confusion_matrix, save_best_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def main():
    print("=== PIPELINE DE EJECUCIÓN ===")
    
    # 1. Cargar y preprocesar datos
    print("\n[1/5] Cargando datos desde el archivo...")
    df_raw = cargar_datos()
    print("\n[2/5] Preprocesando datos...")
    df_preprocessed = preparar_datos(df_raw)


    # 2. Guardar datos procesados
    generar_datos_procesados(df_preprocessed)

    # 3. Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = dividir_datos(df_preprocessed)

    # 3.1 Escalar datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test, preprocessor = escalar_datos(X_train, X_test, y_train, y_test)
    print("\n[4/5] Datos escalados.")
    # 4. Iniciar modelos
    print("\n[4/5] Inicializando modelos...")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    }
    
    # 3. Evaluar y comparar
    print("\n[5/5] Entrenando y evaluando modelos...")
    df_results = evaluar_modelo(models, X_train, y_train, X_test, y_test)
    
    print("\n=== RESULTADOS ===")
    print(df_results.to_string(index=False))
    
    # 4. Exportar matriz de confusión y guardar el mejor modelo
    best_model_name = df_results.iloc[0]['Modelo']
    print(f"\n[4/5] Exportando resultados para el mejor modelo: {best_model_name}")

    # 5. Guardar resultados del mejor modelo
    #save_confusion_matrix(models[best_model_name], X_test, y_test, best_model_name)
    #save_best_model(models[best_model_name], best_model_name)
    

if __name__ == "__main__":
    main()