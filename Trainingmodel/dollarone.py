import pandas as pd
import numpy as np
import glob
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dollarpy import Recognizer, Template, Point

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

def train_1dollar_model(human_files, bot_files):
    templates = []  # List to hold templates for recognizer

    for file_path in human_files:
        df = preprocess_data(load_data(file_path))
        human_points = [Point(row['x_position'], row['y_position']) for _, row in df.iterrows()]
        templates.append(Template('Human', human_points))

    for file_path in bot_files:
        df = preprocess_data(load_data(file_path))
        bot_points = [Point(row['x_position'], row['y_position']) for _, row in df.iterrows()]
        templates.append(Template('Bot', bot_points))

    recognizer = Recognizer(templates)  # Initialize the recognizer with the templates
    return recognizer

def predict_with_1dollar(recognizer, test_file):
    df = preprocess_data(load_data(test_file))
    points = [Point(row['x_position'], row['y_position']) for _, row in df.iterrows()]
    result = recognizer.recognize(points)
    return result

def visualize_data_distribution(human_data, bot_data, test_data):
    print("Before:")
    print(human_data.info())
    print(bot_data.info())
    print(test_data.info())

    human_data = human_data[~human_data.index.duplicated(keep='first')]
    bot_data = bot_data[~bot_data.index.duplicated(keep='first')]
    test_data = test_data[~test_data.index.duplicated(keep='first')]

    fig, axes = plt.subplots(3, 1, figsize=(15, 20))

    sns.scatterplot(data=human_data, x='x_position', y='y_position', ax=axes[0], alpha=0.5, color='blue')
    axes[0].set_title('Human Data: Mouse Position')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')

    sns.scatterplot(data=bot_data, x='x_position', y='y_position', ax=axes[1], alpha=0.5, color='red')
    axes[1].set_title('Bot Data: Mouse Position')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')

    sns.scatterplot(data=test_data, x='x_position', y='y_position', ax=axes[2], alpha=0.5, color='green')
    axes[2].set_title('Test Data: Mouse Position')
    axes[2].set_xlabel('X Position')
    axes[2].set_ylabel('Y Position')

    plt.tight_layout()
    plt.show()

def visualize_movement_speed(human_data, bot_data, test_data):
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    for i, (data, title) in enumerate(zip([human_data, bot_data, test_data], ['Human', 'Bot', 'Test'])):
        sns.histplot(data=data['movement_speed'], ax=axes[i], kde=True)
        axes[i].set_title(f'{title} Data: Movement Speed Distribution')
        axes[i].set_xlabel('Movement Speed')
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def visualize_click_patterns(human_data, bot_data, test_data):
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    for i, (data, title) in enumerate(zip([human_data, bot_data, test_data], ['Human', 'Bot', 'Test'])):
        sns.scatterplot(data=data[data['click'] == 1], x='x_position', y='y_position', ax=axes[i], alpha=0.5)
        axes[i].set_title(f'{title} Data: Click Positions')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')

    plt.tight_layout()
    plt.show()

def main():
    human_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\processed\\*.csv")
    bot_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\botprocessed\\*.csv")

    human_data = pd.concat([preprocess_data(load_data(file)) for file in human_files])
    bot_data = pd.concat([preprocess_data(load_data(file)) for file in bot_files])

    # Visualize data distributions
    test_file = "D:\\AbyssOrder\\Abyss-Order\\datasets2\\test2\\vishalbasker.csv"
    test_data = preprocess_data(load_data(test_file))

    visualize_data_distribution(human_data, bot_data, test_data)
    visualize_movement_speed(human_data, bot_data, test_data)
    visualize_click_patterns(human_data, bot_data, test_data)

    # Train $1 Recognizer model
    recognizer = train_1dollar_model(human_files, bot_files)

    # Predict on a test file
    result = predict_with_1dollar(recognizer, test_file)
    print(f"Predicted: {result[0]}, Confidence: {result[1]:.2f}")

if __name__ == "__main__":
    main()
