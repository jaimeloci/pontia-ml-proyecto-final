
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

ROOT_PATH = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_PATH / "data" / "raw" / "dataset_practica_final.csv"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed" / "dataset_practica_final_preprocessed.csv"
OUTPUTS_DIR = ROOT_PATH / "outputs"
TARGET_COLUMN = 'is_canceled'
IRRELEVANT_COLUMNS = [
    'meal',
    'total_of_special_requests',
    'required_car_parking_spaces',
    'days_in_waiting_list',
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_day_of_month',
    'arrival_date_week_number',
    'reservation_status', 
    'reservation_status_date',
    'assigned_room_type',
    'reservation_status',
    'reservation_status_date', 
    'assigned_room_type', 
    'country' 
]

DICT_MODELS = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

DICT_MODEL_PARAMS = {
    'RandomForestClassifier': {
        #'n_estimators': [50, 100, 200],
        'n_estimators': [50],
        #'max_depth': [8, 12, 16, 25, 30, 35, 40, 45, 50]
        'max_depth': [8, 12, 16]
    },
    'DecisionTreeClassifier': {
        #'max_depth': [8, 12, 16, 25, 30, 35, 40, 45, 50]
        'max_depth': [8, 12, 16]
    },
    'LogisticRegression': {
        # 'C': [0.01, 0.1, 1, 10, 100],
        # 'max_iter': [100, 200, 400, 500, 600, 700, 800, 900, 1000],
        # 'solver': ['liblinear'],
        # 'penalty': ['l1', 'l2'],
         'C': [0.01],
         'max_iter': [100],
         'solver': ['liblinear'],
         'penalty': ['l1', 'l2'],
    }
}