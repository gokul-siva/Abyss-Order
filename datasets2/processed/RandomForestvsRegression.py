import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import glob
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    df['movement_speed'] = np.sqrt(df['x_position'].diff()**2 + df['y_position'].diff()**2) / df['time_diff']
    
    bool_columns = ['button', 'click', 'press']
    for col in bool_columns:
        df[col] = df[col].fillna(0).astype(float).round().astype('int32')
    
    df = pd.get_dummies(df, columns=['key'], prefix='key')
    
    # Feature engineering
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

def select_features(X, y, max_features=20):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=max_features)
    selector.fit(X, y)
    return selector

def build_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )

def evaluate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    y_pred = model.predict(X)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bot', 'human'], yticklabels=['bot', 'human'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def predict_file(model, scaler, feature_selector, file_path, training_columns):
    df = load_and_preprocess_data(file_path)
    features = prepare_features(df, training_columns)
    features = np.clip(features, -1e6, 1e6)
    features_scaled = scaler.transform(features)
    features_selected = feature_selector.transform(features_scaled)
    
    prediction = model.predict(features_selected)
    probabilities = model.predict_proba(features_selected)
    
    return prediction, probabilities

def main():
    human_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\processed\\*.csv")
    bot_files = glob.glob("D:\\AbyssOrder\\Abyss-Order\\datasets2\\botprocessed\\*.csv")
    
    human_data = pd.concat([load_and_preprocess_data(file) for file in human_files])
    bot_data = pd.concat([load_and_preprocess_data(file) for file in bot_files])
    
    all_data = pd.concat([human_data.assign(label=1), bot_data.assign(label=0)]).sample(frac=1, random_state=42)
    
    features = prepare_features(all_data)
    labels = all_data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    
    X_test = np.clip(X_test, -1e6, 1e6)
    X_train = np.clip(X_train, -1e6, 1e6)
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_selector = select_features(X_train_scaled, y_train, max_features=20)
    X_train_selected = feature_selector.transform(X_train_scaled)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    
    rf_model = build_model()
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    print("Random Forest Evaluation:")
    evaluate_model(rf_model, X_test_selected, y_test)
    
    # Compare with a simpler model
    lr_model = LogisticRegression(random_state=42, class_weight='balanced')
    lr_model.fit(X_train_resampled, y_train_resampled)
    
    print("\nLogistic Regression Evaluation:")
    evaluate_model(lr_model, X_test_selected, y_test)
    
    test_file = "D:\\AbyssOrder\\Abyss-Order\\datasets2\\test2\\22227777.csv"
    rf_prediction, rf_probabilities = predict_file(rf_model, scaler, feature_selector, test_file, features.columns)
    lr_prediction, lr_probabilities = predict_file(lr_model, scaler, feature_selector, test_file, features.columns)
    
    print(f"\nRandom Forest: The file {test_file} is predicted to be from a {'HUMAN' if rf_prediction[0] == 1 else 'BOT'}.")
    print(f"Probabilities: Bot: {rf_probabilities[0][0]:.4f}, Human: {rf_probabilities[0][1]:.4f}")
    
    print(f"\nLogistic Regression: The file {test_file} is predicted to be from a {'HUMAN' if lr_prediction[0] == 1 else 'BOT'}.")
    print(f"Probabilities: Bot: {lr_probabilities[0][0]:.4f}, Human: {lr_probabilities[0][1]:.4f}")

if __name__ == "__main__":
    main()