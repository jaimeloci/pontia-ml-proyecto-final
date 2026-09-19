
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from src.config import OUTPUTS_DIR

def evaluate_all_models(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    results = []
    plt.figure(figsize=(10, 8))
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        print(f"Evaluando: {name}...")
        
        if 'Keras' in name:
            model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=0)
            y_pred_proba = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Modelo': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Azar (AUC = 0.500)')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Comparativa de Curvas ROC')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUTS_DIR / "curva_roc_comparativa.png")
    plt.close()

    df_results = pd.DataFrame(results).sort_values(by='F1-score', ascending=False)
    return df_results

def save_confusion_matrix(model, X_test, y_test, model_name: str):
    if 'Keras' in model_name:
        y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Cancelado', 'Cancelado'], yticklabels=['No Cancelado', 'Cancelado'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.savefig(OUTPUTS_DIR / f"matriz_confusion_{model_name.replace(' ', '_').lower()}.png")
    plt.close()