from src.model_trainer import entrenar_modelo
from src.predictor import predecir
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import OUTPUTS_DIR

def evaluar_modelo(models, X_train, y_train, X_test, y_test):

    results = []

    for name, model in models.items():
        model = entrenar_modelo(model, X_train, y_train)
        y_pred = predecir(model, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Modelo': name, 'Precisión': accuracy})

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Precisión', ascending=False).reset_index(drop=True)
    return df_results

def distribucion_variable_objetivo(df, target_column):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    plt.title(f'Distribución de la variable objetivo ({target_column})')
    plt.xticks([0, 1], ['No Cancelado', 'Cancelado'])
    plt.ylabel('Cantidad de reservas')
    plt.savefig(OUTPUTS_DIR / f"distribucion_{target_column}.png")
    plt.close()