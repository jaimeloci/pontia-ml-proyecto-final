
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import RAW_DATA_PATH, TARGET_COLUMN, IRRELEVANT_COLUMNS, PROCESSED_DATA_PATH

def load_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset en bruto."""
    return pd.read_csv(filepath)

def prepare_pipeline_data(test_size: float = 0.2, random_state: int = 42):
    """Limpia los datos y aplica las transformaciones con Train-Test Split."""
    df = load_data()
    
    # 1. Eliminar variables irrelevantes
    df_clean = df.drop(columns=[
        col for col in IRRELEVANT_COLUMNS 
        if col in df.columns
        ])
    
    # 2. Separar características y variable objetivo
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN].astype(int)
    
    # 3. Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Construir transformers
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Guardar el dataset limpio y procesado
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    
    # 5. Fit & Transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    return X_train_prep, X_test_prep, y_train.values, y_test.values, preprocessor