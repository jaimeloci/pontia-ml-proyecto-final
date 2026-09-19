
import os
from pathlib import Path

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