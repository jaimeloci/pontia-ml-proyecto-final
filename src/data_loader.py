from json import encoder

import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, IRRELEVANT_COLUMNS

def cargar_datos():
    return pd.read_csv(RAW_DATA_PATH)

def preparar_datos(df: pd.DataFrame):

    # Quitamos las variables irrelevantes para el modelo, ya que no aportan información útil para la predicción de la variable objetivo.
    df_without_irrelevant = df.drop(columns=IRRELEVANT_COLUMNS)

    # Seteamos la variable objetivo
    target = df_without_irrelevant[TARGET_COLUMN]
    df_preprocessed = df_without_irrelevant.drop(columns=[TARGET_COLUMN])
   
  # Imputamos los valores faltantes en las columnas numéricas
    numeric_columns_with_naan = ['children', 'agent', 'company']
    imputer_numerico = SimpleImputer(strategy="mean")
    df_preprocessed[numeric_columns_with_naan] = imputer_numerico.fit_transform(df_preprocessed[numeric_columns_with_naan])
    
    # Codificación de variables categóricas
    categorical_columns = df_preprocessed.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    df_encoded = pd.DataFrame(
        encoder.fit_transform(df_preprocessed[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df_encoded.columns = df_encoded.columns.str.lower()
    df_preprocessed = pd.concat([df_preprocessed.drop(columns=categorical_columns), df_encoded], axis=1)

    # Escalado de variables numéricas
    scaler = StandardScaler()
    df_preprocessed[df_preprocessed.columns] = scaler.fit_transform(df_preprocessed[df_preprocessed.columns])

    df_preprocessed[TARGET_COLUMN] = target.values

    # Reemplzamos los valores numéricos de la columna 'target' por los nombres de las clases
    dict_canceled = {
        'No cancelado' : 0,
        'Cancelado' : 1
    }

    df_preprocessed[TARGET_COLUMN] = df_preprocessed[TARGET_COLUMN].replace(dict_canceled)

    return df_preprocessed

def generar_csv_datos_preprocesados(df: pd.DataFrame):
    df_preprocessed = df
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_preprocessed.to_csv(PROCESSED_DATA_PATH, index=False)

