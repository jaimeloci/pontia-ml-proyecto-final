import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMN

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
      return X_train, X_test, y_train, y_test, scaler

def entrenar_modelo(model, X_train, y_train):
    # Entrenamos un modelo usando GridSearchCV para encontrar los mejores hiperparámetros
    dict_parametros = get_model_params(model)
    model = GridSearchCV(model, dict_parametros, cv=3, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)
    # Mostramos los mejores hiperparámetros encontrados
    print(f"Mejores hiperparámetros encontrados: {model.best_params_}")
    print(f"Mejor score obtenido: {model.best_score_:.2%}")

    return model

def get_model_params(model):
    switcher = {
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 12, 16, 25, 30, 35, 40, 45, 50]
        },
        'DecisionTreeClassifier': {
            'max_depth': [8, 12, 16, 25, 30, 35, 40, 45, 50]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200, 400, 500, 600, 700, 800, 900, 1000],
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
        }
    }
    return switcher.get(model.__class__.__name__, {})