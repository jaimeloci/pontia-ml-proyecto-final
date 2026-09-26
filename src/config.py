
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

ROOT_PATH = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_PATH / "data" / "raw" / "dataset_practica_final.csv"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed" / "dataset_practica_final_preprocessed.csv"
OUTPUTS_DIR = ROOT_PATH / "outputs"
TARGET_COLUMN = 'is_canceled'
IRRELEVANT_COLUMNS = [
    'assigned_room_type',
    'booking_changes',
    'reservation_status',
    'reservation_status_date'
]

DICT_CAST_CATEGORY_COLS = {
    'arrival_date_month': 'str',
    'agent': 'str',
    'company': 'str',
    'is_repeated_guest': 'str',
}

DICT_MODELS = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

DICT_GRID_PARAMS = {
    'Random Forest': {
        'model__max_depth': [4, 8, 12],
        'model__n_estimators': [10, 20, 30],
        'model__min_samples_leaf': [1, 2, 4],
        'model__min_samples_split': [2, 5, 8],
    },
    'Decision Tree': {
        'model__max_depth': [8, 12, 16]
    },
    'Logistic Regression': {
         'model__C': [0.01],
         'model__max_iter': [10],
         'model__solver': ['liblinear'],
         'model__penalty': ['l1', 'l2'],
    },
    'LightGBM': {
        'prep__cat__min_frequency': [5, 10, 20],
        'model__n_estimators': [200, 500],
        'model__num_leaves': [31, 63, 127],
        'model__learning_rate': [0.05, 0.1],
        'model__min_child_samples': [20, 50],
        'model__subsample': [0.8, 1.0],
        'model__subsample_freq': [1],
        'model__colsample_bytree': [0.8, 1.0],
    },
    'Gradient Boosting': {
        'model__n_estimators': [10, 20],
        'model__learning_rate': [0.03, 0.1],
        'model__max_depth': [3, 5, 7]
    }
}