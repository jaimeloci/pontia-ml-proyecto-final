
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

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
    'arrival_date_month': 'object',
    'agent': 'object',
    'company': 'object',
    'is_repeated_guest': 'object',
}

DICT_MODELS = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, max_depth=-1, n_jobs=1),
}

DICT_COLUMN_TRANSFORMER_PARAMS = {
    'Logistic Regression': {
        'num': StandardScaler(),
        'cat': OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    },
    'Decision Tree': {
        'num': 'passthrough',
        'cat': OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    },
    'Random Forest': {
        'num': 'passthrough',
        'cat': OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    },
    'LightGBM': {
        'num': 'passthrough',
        'cat': OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    },
    'Red Neuronal Keras': {
        'num': StandardScaler(),
        'cat': OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=20)
    },
}

DICT_GRID_PARAMS = {
    'Logistic Regression': {
        'prep__cat__min_frequency': [10, 20],
        'model__C': [1, 10],
        'model__max_iter': [50, 100],
        'model__solver': ['liblinear'],
    },
    'Decision Tree': {
        'prep__cat__min_frequency': [10, 20],
        'model__max_depth': [5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
    },
    'Random Forest': {
        'prep__cat__min_frequency': [10, 20],
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
    },
    'LightGBM': {
        'prep__cat__min_frequency': [10, 20],
        'model__n_estimators': [200, 500],
        'model__num_leaves': [63, 127],
        'model__learning_rate': [0.05, 0.1],
        'model__min_child_samples': [20, 50],
        'model__subsample': [0.8, 1.0],
        'model__subsample_freq': [1],
        'model__colsample_bytree': [0.8, 1.0],
    },
}

DICT_GRID_PARAMS_SIMPLIFIED = {
    'Logistic Regression': {
        'prep__cat__min_frequency': [20],
        'model__C': [10],
        'model__max_iter': [50],
        'model__solver': ['liblinear'],
    },
    'Decision Tree': {
        'prep__cat__min_frequency': [10],
        'model__max_depth': [20],
        'model__min_samples_split': [5],
        'model__min_samples_leaf': [1],
    },
    'Random Forest': {
        'prep__cat__min_frequency': [20],
        'model__n_estimators': [200],
        'model__max_depth': [None],
        'model__min_samples_split': [2],
        'model__min_samples_leaf': [1],
    },
    'LightGBM': {
        'prep__cat__min_frequency': [10],
        'model__n_estimators': [500],
        'model__num_leaves': [127],
        'model__learning_rate': [0.1],
        'model__min_child_samples': [20],
        'model__subsample': [0.8],
        'model__subsample_freq': [1],
        'model__colsample_bytree': [1.0],
    },
}