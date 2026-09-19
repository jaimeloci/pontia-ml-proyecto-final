from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.data_loader import prepare_pipeline_data
from src.hyperparameter_tuning import optimize_decision_tree, optimize_random_forest
from src.evaluator import evaluate_all_models, save_confusion_matrix
from src.predictor import save_best_model

def build_keras_model(input_dim: int):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("=== PIPELINE DE EJECUCIÓN ===")
    
    # 1. Cargar y preprocesar datos
    print("\n[1/4] Procesando datos...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_pipeline_data()
    
    # 2. Definir modelos y aplicar Bonus de Optimización
    print("\n[2/4] Optimizando e inicializando modelos...")
    best_rf = optimize_random_forest(X_train, y_train)
    best_dt = optimize_decision_tree(X_train, y_train)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Decision Tree (Optimizado)': best_dt,
        'Random Forest (Optimizado)': best_rf,
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Deep Neural Network (Keras)': build_keras_model(X_train.shape[1])
    }
    
    # 3. Evaluar y comparar
    print("\n[3/4] Entrenando y evaluando modelos...")
    df_results = evaluate_all_models(models, X_train, y_train, X_test, y_test)
    
    print("\n=== RESULTADOS ===")
    print(df_results.to_string(index=False))
    
    # 4. Exportar matriz de confusión y guardar el mejor modelo
    best_model_name = df_results.iloc[0]['Modelo']
    print(f"\n[4/4] Exportando resultados para el mejor modelo: {best_model_name}")
    save_confusion_matrix(models[best_model_name], X_test, y_test, best_model_name)
    save_best_model(models[best_model_name], best_model_name)

if __name__ == "__main__":
    main()