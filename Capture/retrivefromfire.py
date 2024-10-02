import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

#use \\ in the path
path_to_json="C:\\Users\\sripr\\Downloads\\mouse2772-9e216-firebase-adminsdk-o884w-97a8cd3c97.json"


# Initialize Firebase Admin SDK with service account
cred = credentials.Certificate(path_to_json)  # Replace with your service account key file path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mouse2772-9e216-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase database URL
})

def fetch_and_save_user_data():
   
    ref = db.reference('/')
    users = ref.get()

    

    if users:
        print("Retrieving user data...")
        for user_name, user_data in users.items():
            
            rows = []
            if isinstance(user_data, dict):  
                for timestamp, data in user_data.items():
                    row = {
                        'timestamp': timestamp,
                        'x_position': data.get('x_position', None),
                        'y_position': data.get('y_position', None),
                        'button': data.get('button', None),
                        'click': data.get('click', None),
                        'dx': data.get('dx', None),
                        'dy': data.get('dy', None),
                        'key': data.get('key', None),
                        'press': data.get('press', None)
                    }
                    rows.append(row)
            elif isinstance(user_data, list):  
                for item in user_data:
                    
                    if isinstance(item, dict):
                        row = {
                            'timestamp': item.get('timestamp', None),
                            'x_position': item.get('x_position', None),
                            'y_position': item.get('y_position', None),
                            'button': item.get('button', None),
                            'click': item.get('click', None),
                            'dx': item.get('dx', None),
                            'dy': item.get('dy', None),
                            'key': item.get('key', None),
                            'press': item.get('press', None)
                        }
                        rows.append(row)

            
            df = pd.DataFrame(rows)
            file_name = f"{user_name}.csv" 
            df.to_csv(file_name, index=False)
            print(f"Data for {user_name} saved to {file_name}.")
    else:
        print("No user data found in the database.")

if __name__ == "__main__":
    fetch_and_save_user_data()
