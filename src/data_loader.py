import pandas as pd
import os
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, IRRELEVANT_COLUMNS

def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)

def preparar_datos_old(df: pd.DataFrame) -> pd.DataFrame:
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

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina variables irrelevantes y asegura el formato correcto del target.
    El escalado, imputación y codificación se delegan al ColumnTransformer dentro del Pipeline.
    """
    # Eliminar variables irrelevantes
    df_clean = df.drop(columns=[col for col in IRRELEVANT_COLUMNS if col in df.columns])

    # 'agent' y 'company': cambiamos nulos por ceros
    for col in ['agent', 'company']:
        df_clean[col] = df_clean[col].fillna(0).astype(int)

    # 'children': sustituir el valor nulo por cero.
    df_clean['children'] = df_clean['children'].fillna(0).astype(int)

    # 'country': sustituir nulos por el valor 'unknown'.
    df_clean['country'] = df_clean['country'].fillna('unknown')

    # 'arrival_date_month': sustituir valor textual por ordinal.
    df_clean['arrival_date_month'] = df_clean['arrival_date_month'].replace({
        'January': '1',
        'February': '2',
        'March': '3',
        'April': '4',
        'May': '5',
        'June': '6',
        'July': '7',
        'August': '8',
        'September': '9',
        'October': '10',
        'November': '11',
        'December': '12'
    }).astype(int)

    # Eliminar outliers en 'adr'
    df_clean = df_clean[df_clean['adr'] != 5400]
    df_clean = df_clean[df_clean['adr'] != -6.38]

    return df_clean

def generar_csv_datos_preprocesados(df: pd.DataFrame):
    df_clean = df
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

