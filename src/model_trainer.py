import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import TARGET_COLUMN, DICT_COLUMN_TRANSFORMER_PARAMS, DICT_GRID_PARAMS, DICT_GRID_PARAMS_SIMPLIFIED

def dividir_datos(df: pd.DataFrame, stratify: bool = True):
    """
    Divide los datos en train/test manteniendo estratificación opcional[cite: 1].
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    stratify_col = y if stratify else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_col)

def construir_preprocesador(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    """
    Construye el ColumnTransformer
    """

    if model_name not in DICT_COLUMN_TRANSFORMER_PARAMS:
        raise ValueError(
            f"Modelo no soportado: {model_name!r}. "
            f"Opciones válidas: {list(DICT_COLUMN_TRANSFORMER_PARAMS)}"
        )
    
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
    
    column_transformer_params = DICT_COLUMN_TRANSFORMER_PARAMS.get(model_name, {})

    preprocessor = ColumnTransformer([
        ('num', column_transformer_params['num'], numerical_columns),
        ('cat', column_transformer_params['cat'], categorical_columns)
    ])

    if model_name == 'LightGBM':
        preprocessor = preprocessor.set_output(transform='pandas') # Para que a LightGBM le lleguen los nombre de las columnas

    return preprocessor

def entrenar_modelos(models_dict: dict, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5):
    trained_pipelines = {}
    
    for name, model_inst in models_dict.items():
        print(f"--- Entrenando y buscando hiperparámetros: {name} ---")
        
        # Se ensambla el Pipeline completo con ColumnTransformer + Modelo
        preprocessor = construir_preprocesador(X=X_train, model_name=name)
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', model_inst)
        ])
        
        param_grid = DICT_GRID_PARAMS.get(name, {})
        # param_grid = DICT_GRID_PARAMS_SIMPLIFIED.get(name, {})
        
        grid_search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',  # Ajustado al criterio del cuaderno
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Mejores parámetros para {name}: {grid_search.best_params_}")
        trained_pipelines[name] = grid_search.best_estimator_
        
    return trained_pipelines