import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import datetime

def load_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def parse_timestamp(timestamp_str):
    try:
        return pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # If the full datetime parsing fails, try parsing just the time part
        try:
            time_part = pd.to_datetime(timestamp_str, format='%M:%S.%f').time()
            # Combine with a dummy date
            return datetime.datetime.combine(datetime.date(1970, 1, 1), time_part)
        except ValueError:
            # If all parsing attempts fail, return NaT (Not a Time)
            return pd.NaT
        
def preprocess_data(df):
    # Convert timestamp to datetime
    df['timestamp'] = df['timestamp'].apply(lambda x: parse_timestamp(x))
    
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    
    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    # Calculate movement speed
    df['movement_speed'] = np.sqrt(df['x_position'].diff()**2 + df['y_position'].diff()**2) / df['time_diff']
    
    # Convert boolean columns to int, handling NaN values
    bool_columns = ['button', 'click', 'press']
    for col in bool_columns:
        df[col] = df[col].fillna(0).astype(float).round().astype('int32')
    
    
    
    # One-hot encode the 'key' column
    df = pd.get_dummies(df, columns=['key'], prefix='key')
    
    print(df.head())
    return df

def engineer_features(df):
    # Calculate statistics for numerical columns
    num_columns = ['x_position', 'y_position', 'movement_speed']
    for col in num_columns:
        df[f'{col}_mean'] = df[col].rolling(window=10).mean()
        df[f'{col}_std'] = df[col].rolling(window=10).std()
    
    # Calculate click and key press frequencies
    df['click_frequency'] = df['click'].rolling(window=100).mean()
    df['key_press_frequency'] = df['press'].rolling(window=100).mean()
    
    # Calculate time since last click and key press
    df['time_since_last_click'] = df['timestamp'].where(df['click'] == 1).ffill().fillna(df['timestamp'].iloc[0])
    df['time_since_last_click'] = (df['timestamp'] - df['time_since_last_click']).dt.total_seconds()
    
    df['time_since_last_key_press'] = df['timestamp'].where(df['press'] == 1).ffill().fillna(df['timestamp'].iloc[0])
    df['time_since_last_key_press'] = (df['timestamp'] - df['time_since_last_key_press']).dt.total_seconds()
    
    return df

def prepare_features(df):
    # Drop unnecessary columns
    columns_to_drop = ['timestamp']
    features = df.drop(columns=columns_to_drop)
    
    # Handle missing values
    features = features.fillna(0)
    
    return features

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    # Load data
    human_data = load_data("D:\\AbyssOrder\\Abyss-Order\\datasets\\human\\")
    bot_data = load_data("D:\\AbyssOrder\\Abyss-Order\\datasets\\bot\\")
    
    # Add labels
    human_data['label'] = 'human'
    bot_data['label'] = 'bot'
    
    # Combine datasets
    all_data = pd.concat([human_data, bot_data], ignore_index=True)
    
    # Preprocess and engineer features
    all_data = preprocess_data(all_data)
    all_data = engineer_features(all_data)
    
    # Prepare features and labels
    features = prepare_features(all_data)
    labels = all_data['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Function to predict on new data
    def predict_new_data(csv_file):
        new_data = pd.read_csv(csv_file)
        new_data = preprocess_data(new_data)
        new_data = engineer_features(new_data)
        new_features = prepare_features(new_data)
        new_features_scaled = scaler.transform(new_features)
        prediction = model.predict(new_features_scaled)
        return "Human" if prediction[0] == 'human' else "Bot"
    
    # Example usage:
    result = predict_new_data("path/to/new/data.csv")
    print(f"The new data is predicted to be: {result}")

if __name__ == "__main__":
    main()