import pandas as pd
import os
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, IRRELEVANT_COLUMNS

def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina variables irrelevantes y asegura el formato correcto del target.
    El escalado, imputación y codificación se delegan al ColumnTransformer dentro del Pipeline.
    """
    # Eliminar variables irrelevantes
    df_clean = df.drop(columns=[col for col in IRRELEVANT_COLUMNS if col in df.columns])

    # Asegurar mapeo de la variable objetivo a binario (0 y 1)[cite: 1]
    if df_clean[TARGET_COLUMN].dtype == 'object':
        dict_canceled = {
            'No cancelado': 0,
            'Cancelado': 1
        }
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map(dict_canceled)

    return df_clean

def generar_csv_datos_preprocesados(df: pd.DataFrame):
    df_preprocessed = df
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_preprocessed.to_csv(PROCESSED_DATA_PATH, index=False)

