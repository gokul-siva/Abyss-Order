import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
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
    df['timestamp'] = df['timestamp'].apply(parse_timestamp)
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
        features = features[training_columns]

    features = features.fillna(0)
    return features

def select_features(X, y):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=50)
    selector.fit(X, y)
    return selector

def build_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bot', 'human'], yticklabels=['bot', 'human'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def predict_file(model, scaler, feature_selector, file_path, training_columns):
    df = load_data(file_path)
    df = preprocess_data(df)
    df = engineer_features(df)
    
    features = prepare_features(df, training_columns)
    features = np.clip(features, -1e6, 1e6)
    features_scaled = scaler.transform(features)
    features_selected = feature_selector.transform(features_scaled)
    
    prediction = model.predict(features_selected)
    
    return prediction

def main():
    human_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\processed\\*.csv")
    bot_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\botprocessed\\*.csv")
    
    human_data = pd.concat([preprocess_data(load_data(file)) for file in human_files])
    bot_data = pd.concat([preprocess_data(load_data(file)) for file in bot_files])
    
    all_data = pd.concat([human_data.assign(label=1), bot_data.assign(label=0)]).sample(frac=1, random_state=42)
    
    features = prepare_features(all_data)
    labels = all_data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    X_test = np.clip(X_test, -1e6, 1e6)
    X_train = np.clip(X_train, -1e6, 1e6)
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_selector = select_features(X_train_scaled, y_train)
    X_train_selected = feature_selector.transform(X_train_scaled)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    
    model = build_model(X_train_resampled, y_train_resampled)
    
    evaluate_model(model, X_test_selected, y_test)
    
    test_file = "D:\\AbyssOrder\\Abyss-Order\\datasets2\\test2\\22227777.csv"
    prediction = predict_file(model, scaler, feature_selector, test_file, features.columns)
    
    print(f"\nThe file {test_file} is predicted to be from a {'HUMAN' if prediction[0] == 1 else 'BOT'}.")

if __name__ == "__main__":
    main()