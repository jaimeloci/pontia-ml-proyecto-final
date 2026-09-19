import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMN

def dividir_datos(df_preprocessed: pd.DataFrame):
        X = df_preprocessed.drop(columns=[TARGET_COLUMN])
        y = df_preprocessed[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

def escalar_datos(X_train, X_test, y_train, y_test, ):
    # Escalar características
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)
      return X_train, X_test, y_train, y_test, scaler

def entrenar_modelo(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model