import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import TARGET_COLUMN, DICT_GRID_PARAMS

def dividir_datos(df: pd.DataFrame, stratify: bool = True):
    """
    Divide los datos en train/test manteniendo estratificación opcional[cite: 1].
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    stratify_col = y if stratify else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_col)

def construir_preprocesador(X: pd.DataFrame, min_frequency: int = 10) -> ColumnTransformer:
    """
    Construye el ColumnTransformer centralizando imputación, escalado y encoding[cite: 1].
    """
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()

    # Pipeline para numéricas: Imputación por media + Escalado[cite: 1]
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para categóricas: One-Hot Encoding manejando categorías infrecuentes[cite: 1]
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=min_frequency))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_columns),
        ('cat', cat_pipeline, categorical_columns)
    ])

    return preprocessor

def entrenar_modelos(models_dict: dict, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 3):
    trained_pipelines = {}
    
    for name, model_inst in models_dict.items():
        print(f"--- Entrenando y buscando hiperparámetros: {name} ---")
        
        # Se ensambla el Pipeline completo con ColumnTransformer + Modelo[cite: 1]
        preprocessor = construir_preprocesador(X_train)
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', model_inst)
        ])
        
        param_grid = DICT_GRID_PARAMS.get(name, {})
        
        grid_search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',  # Ajustado al criterio del cuaderno[cite: 1]
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Mejores parámetros para {name}: {grid_search.best_params_}")
        trained_pipelines[name] = grid_search.best_estimator_
        
    return trained_pipelines