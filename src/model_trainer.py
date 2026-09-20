import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DICT_MODEL_PARAMS, TARGET_COLUMN

def dividir_datos(df_preprocessed: pd.DataFrame): 
    # Eliminamos la columna is_canceled de las variables independientes
    X = df_preprocessed.drop(columns=[TARGET_COLUMN])
    y = df_preprocessed[TARGET_COLUMN]

    # Hacemos un train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def escalar_datos(X_train, X_test, y_train, y_test, ):
    # Escalar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def entrenar_modelos(models, X_train, y_train):
    # Entrenamos los modelos usando GridSearchCV para encontrar los mejores hiperparámetros
    for model_name, model in models.items():

        dict_parametros = get_model_params(model) # Obtenemos los parámetros segun el algoritmo del modelo
        model = GridSearchCV(model, dict_parametros, cv=3, scoring='accuracy', n_jobs=-1, refit=True)

        model.fit(X_train, y_train)

        # Mostramos los mejores hiperparámetros encontrados
        print(f"Modelo: {model_name}")
        print(f"Mejores hiperparámetros encontrados: {model.best_params_}")
        print(f"Mejor score obtenido: {model.best_score_:.2%}")
        print(f"==============================")

        # Guardamos directamente el modelo óptimo (GridSearchCV ya lo reentrenó(refit) sobre X_train)
        models[model_name] = model.best_estimator_

    return models

def get_model_params(model):
    switcher = DICT_MODEL_PARAMS
    return switcher.get(model.__class__.__name__, {})