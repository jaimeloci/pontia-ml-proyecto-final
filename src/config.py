
from pathlib import Path

# Directorio raíz del proyecto (pontia-ml-proyecto-final)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "dataset_practica_final.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "dataset_practica_final_preprocessed.csv"

# Rutas de salida y modelos
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Parámetros del problema
TARGET_COLUMN = "is_canceled"
IRRELEVANT_COLUMNS = [
    'meal', 'total_of_special_requests', 'required_car_parking_spaces', 
    'days_in_waiting_list', 'arrival_date_year', 'arrival_date_month', 
    'arrival_date_day_of_month', 'arrival_date_week_number', 
    'reservation_status', 'reservation_status_date', 'assigned_room_type', 'country'
]