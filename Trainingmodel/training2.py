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
        try:
            time_part = pd.to_datetime(timestamp_str, format='%M:%S.%f').time()
            return datetime.datetime.combine(datetime.date(1970, 1, 1), time_part)
        except ValueError:
            return pd.NaT

def preprocess_data(df):
    df['timestamp'] = df['timestamp'].apply(lambda x: parse_timestamp(x))
    
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    df['movement_speed'] = np.sqrt(df['x_position'].diff()**2 + df['y_position'].diff()**2) / df['time_diff']
    
    bool_columns = ['button', 'click', 'press']
    for col in bool_columns:
        df[col] = df[col].fillna(0).astype(float).round().astype('int32')
    
    df = pd.get_dummies(df, columns=['key'], prefix='key')
    
    return df

def engineer_features(df):
    num_columns = ['x_position', 'y_position', 'movement_speed']
    for col in num_columns:
        df[f'{col}_mean'] = df[col].rolling(window=10).mean()
        df[f'{col}_std'] = df[col].rolling(window=10).std()
    
    df['click_frequency'] = df['click'].rolling(window=100).mean()
    df['key_press_frequency'] = df['press'].rolling(window=100).mean()
    
    df['time_since_last_click'] = df['timestamp'].where(df['click'] == 1).ffill().fillna(df['timestamp'].iloc[0])
    df['time_since_last_click'] = (df['timestamp'] - df['time_since_last_click']).dt.total_seconds()
    
    df['time_since_last_key_press'] = df['timestamp'].where(df['press'] == 1).ffill().fillna(df['timestamp'].iloc[0])
    df['time_since_last_key_press'] = (df['timestamp'] - df['time_since_last_key_press']).dt.total_seconds()
    
    return df

def prepare_features(df):
    columns_to_drop = ['timestamp', 'label']
    features = df.drop(columns=columns_to_drop)
    
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
    human_data = load_data("D:\\AbyssOrder\\Abyss-Order\\datasets\\human\\")
    bot_data = load_data("D:\\AbyssOrder\\Abyss-Order\\datasets\\bot\\")
    
    human_data['label'] = 1  # Label humans as 1
    bot_data['label'] = 0    # Label bots as 0
    
    all_data = pd.concat([human_data, bot_data], ignore_index=True)
    
    all_data = preprocess_data(all_data)
    all_data = engineer_features(all_data)
    
    features = prepare_features(all_data)
    labels = all_data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    if np.any(np.isinf(X_train)) or np.any(np.isnan(X_train)):
        print("X_train contains infinity or NaN values")
    # Optionally, handle these values (e.g., replace them with a finite value or remove the problematic rows)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e9, neginf=-1e9)
    print(f"Max value in X_train: {np.max(X_train)}")
    print(f"Min value in X_train: {np.min(X_train)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = train_model(X_train_scaled, y_train)
    
    evaluate_model(model, X_test_scaled, y_test)
    
    def predict_new_data(csv_file):
        new_data = pd.read_csv(csv_file)
        new_data = preprocess_data(new_data)
        new_data = engineer_features(new_data)
        new_features = prepare_features(new_data)
        new_features_scaled = scaler.transform(new_features)
        prediction = model.predict(new_features_scaled)
        return "Human" if prediction[0] == 1 else "Bot"
    
    result = predict_new_data("D:\\AbyssOrder\\Abyss-Order\\datasets\\test\\vishalbasker.csv")
    print(f"The new data is predicted to be: {result}")

if __name__ == "__main__":
    main()
