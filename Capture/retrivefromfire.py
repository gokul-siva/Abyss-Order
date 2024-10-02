import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# Initialize Firebase Admin SDK with service account
cred = credentials.Certificate("C:\\Users\\sripr\\Downloads\\mouse2772-9e216-firebase-adminsdk-o884w-97a8cd3c97.json")  # Replace with your service account key file path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mouse2772-9e216-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase database URL
})

def fetch_and_save_user_data():
    # Get a reference to the root of the database
    ref = db.reference('/')
    users = ref.get()  # Fetch all user data

    # Debugging: print the structure of users
    print("Structure of users:", users)

    if users:
        print("Retrieving user data...")
        for user_name, user_data in users.items():
            # Prepare the data for saving to CSV
            rows = []
            if isinstance(user_data, dict):  # Check if user_data is a dictionary
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
            elif isinstance(user_data, list):  # Handle case where user_data is a list
                for item in user_data:
                    # Assuming item is a dictionary-like structure
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

            # Create a DataFrame and save to CSV
            df = pd.DataFrame(rows)
            file_name = f"{user_name}.csv"  # CSV file name based on user name
            df.to_csv(file_name, index=False)
            print(f"Data for {user_name} saved to {file_name}.")
    else:
        print("No user data found in the database.")

if __name__ == "__main__":
    fetch_and_save_user_data()
