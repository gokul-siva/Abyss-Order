import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

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

def prepare_features(df, training_columns=None):
    columns_to_drop = ['timestamp']
    if 'label' in df.columns:
        columns_to_drop.append('label')
    features = df.drop(columns=columns_to_drop)
    
    if training_columns is not None:
        missing_cols = set(training_columns) - set(features.columns)
        for col in missing_cols:
            features[col] = 0
        features = features[training_columns]  # Reorder to match training set

    features = features.fillna(0)
    return features

def visualize_data(df, label):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['x_position'], label='X Position')
    plt.plot(df['timestamp'], df['y_position'], label='Y Position', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title(f'{label} Movement Over Time')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['movement_speed'], label='Movement Speed', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_model_incrementally(model, scaler, file_path, label, training_columns=None):
    df = load_data(file_path)
    df['label'] = label  # Assign label: 1 for human, 0 for bot
    df = preprocess_data(df)
    df = engineer_features(df)
    
    # Visualize time-dependent data
    visualize_data(df, "Human" if label == 1 else "Bot")
    
    features = prepare_features(df, training_columns)
    
    # Optionally handle infinity and large values by capping
    features = np.clip(features, -1e6, 1e6)
    
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    
    labels = df['label']
    
    # Incrementally train the model
    model.fit(features_scaled, labels)
    
    return model, scaler, list(features.columns)  # Return updated model and scaler

def evaluate_model(model, X_test):
    predictions = model.predict(X_test)
    
    # Output the predictions
    for i, pred in enumerate(predictions):
        if pred == 1:
            print(f"Row {i+1} is predicted as: Human")
        else:
            print(f"Row {i+1} is predicted as: Bot")

def main():
    human_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets\\human\\*.csv")
    bot_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets\\bot\\*.csv")
    
    # Initialize model and scaler
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = None
    training_columns = None
    
    # Train on human files incrementally
    for file in human_files:
        model, scaler, training_columns = train_model_incrementally(model, scaler, file, label=1, training_columns=training_columns)
    
    # Train on bot files incrementally
    for file in bot_files:
        model, scaler, training_columns = train_model_incrementally(model, scaler, file, label=0, training_columns=training_columns)
    
    # Now, predict on new unseen data (test data without label)
    test_data = load_data("D:\\AbyssOrder\\Abyss-Order\\datasets\\test\\vishalbasker.csv")
    test_data = preprocess_data(test_data)
    test_data = engineer_features(test_data)
    X_test = prepare_features(test_data, training_columns=training_columns)
    X_test = np.clip(X_test, -1e6, 1e6)
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate the model with predictions (without y_test since we're predicting)
    evaluate_model(model, X_test_scaled)

if __name__ == "__main__":
    main()
