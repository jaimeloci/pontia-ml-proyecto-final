from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def optimize_random_forest(X_train, y_train):
    """Busca los mejores hiperparámetros para Random Forest."""
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Mejores parámetros para Random Forest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def optimize_decision_tree(X_train, y_train):
    """Busca los mejores hiperparámetros para Decision Tree."""
    param_grid = {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Mejores parámetros para Decision Tree: {grid_search.best_params_}")
    return grid_search.best_estimator_

    