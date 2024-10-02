import math
import csv
import firebase_admin
from firebase_admin import credentials, db

def is_json_compliant(value):
    # Check if the value is a valid JSON number
    if isinstance(value, float):
        return not (math.isnan(value) or math.isinf(value))
    return True

def sanitize_value(value):
    try:
        float_val = float(value)
        if is_json_compliant(float_val):
            return float_val
        else:
            return 0.0  # Replace invalid float values with 0.0
    except ValueError:
        return value  # If not a float, return as is
    
cred = credentials.Certificate("C:\\Users\\sripr\\Downloads\\mouse2772-9e216-firebase-adminsdk-o884w-97a8cd3c97.json")  # Replace with your service account key file path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mouse2772-9e216-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase database URL
})

def upload_to_firebase(csv_file):
    ref = db.reference('mouse_data')
    with open(csv_file, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sanitized_row = {key: sanitize_value(value) for key, value in row.items()}
            print(sanitized_row)
            ref.push(sanitized_row)

upload_to_firebase("send.csv")