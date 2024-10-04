import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

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

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_classes, y_pred_classes))

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bot', 'human'], yticklabels=['bot', 'human'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def predict_file(model, scaler, file_path, training_columns, time_steps):
    df = load_data(file_path)
    df = preprocess_data(df)
    df = engineer_features(df)
    
    features = prepare_features(df, training_columns)
    features = np.clip(features, -1e6, 1e6)
    features_scaled = scaler.transform(features)
    
    X_seq, _ = create_sequences(features_scaled, np.zeros(len(features_scaled)), time_steps)
    predictions = model.predict(X_seq)
    
    avg_probability = np.mean(predictions, axis=0)
    majority_vote = np.argmax(avg_probability)
    
    return majority_vote, avg_probability

def visualize_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    human_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\processed\\*.csv")
    bot_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\botprocessed\\*.csv")
    
    human_data = pd.concat([preprocess_data(load_data(file)) for file in human_files])
    bot_data = pd.concat([preprocess_data(load_data(file)) for file in bot_files])
    
    all_data = pd.concat([human_data.assign(label=1), bot_data.assign(label=0)]).sample(frac=1, random_state=42)
    
    features = prepare_features(all_data)
    labels = all_data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_test = np.clip(X_test, -1e6, 1e6)
    X_train = np.clip(X_train, -1e6, 1e6)
    X_val=np.clip(X_val, -1e6, 1e6)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    time_steps = 10
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.reset_index(drop=True), time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.reset_index(drop=True), time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.reset_index(drop=True), time_steps)
    
    y_train_cat = to_categorical(y_train_seq)
    y_val_cat = to_categorical(y_val_seq)
    y_test_cat = to_categorical(y_test_seq)
    
    model, history = train_model(X_train_seq, y_train_cat, X_val_seq, y_val_cat)
    
    visualize_training_history(history)
    
    evaluate_model(model, X_test_seq, y_test_cat)
    
    test_file = "D:\\AbyssOrder\\Abyss-Order\\datasets2\\test2\\vishalbasker.csv"
    prediction, probabilities = predict_file(model, scaler, test_file, features.columns, time_steps)
    
    print(f"\nThe file {test_file} is predicted to be from a {'HUMAN' if prediction == 1 else 'BOT'}.")
    print(f"Probability of being a bot: {probabilities[0]:.2f}")
    print(f"Probability of being a human: {probabilities[1]:.2f}")

if __name__ == "__main__":
    main()