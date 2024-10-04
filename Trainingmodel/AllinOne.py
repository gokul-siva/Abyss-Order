import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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

def build_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )

def build_isolation_forest():
    return IsolationForest(contamination=0.1, random_state=42)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

class OneDollarRecognizer:
    def __init__(self):
        self.templates = []

    def add_template(self, points, label):
        self.templates.append((points, label))

    def recognize(self, points):
        best_distance = float('inf')
        best_template = None
        for template, label in self.templates:
            distance = self.distance_at_best_angle(points, template)
            if distance < best_distance:
                best_distance = distance
                best_template = label
        return best_template

    def distance_at_best_angle(self, points1, points2):
        # Simplified implementation
        points1 = np.array(points1)
        points2 = np.array(points2)
        
        # If points1 or points2 are 1D, convert them to 2D with a single row
        if points1.ndim == 1:
            points1 = points1[np.newaxis, :]
        if points2.ndim == 1:
            points2 = points2[np.newaxis, :]
        
        # Calculate the Euclidean distance
        return np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=1)))

def evaluate_model(model, X, y, model_name):
    if model_name == 'LSTM':
        # Reshape data for LSTM
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        scores = model.evaluate(X, y, verbose=0)
        print(f"{model_name} Test Accuracy: {scores[1]:.3f}")
        return
    elif model_name == 'Isolation Forest':
        y_pred = model.predict(X)
        y_pred = np.where(y_pred == 1, 0, 1)  # Invert predictions
    elif model_name == 'OneDollar':
        y_pred = [model.recognize(x) for x in X]
    else:
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"{model_name} Cross-validation scores: {cv_scores}")
        print(f"{model_name} Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        y_pred = model.predict(X)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y, y_pred))
    print(f"\n{model_name} Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bot', 'human'], yticklabels=['bot', 'human'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

def predict_file(models, scaler, feature_selector, file_path, training_columns):
    df = load_and_preprocess_data(file_path)
    features = prepare_features(df, training_columns)
    features = np.clip(features, -1e6, 1e6)
    features_scaled = scaler.transform(features)
    features_selected = feature_selector.transform(features_scaled)

    results = {}
    for name, model in models.items():
        if name == 'LSTM':
            features_reshaped = features_selected.reshape((features_selected.shape[0], 1, features_selected.shape[1]))
            prediction = (model.predict(features_reshaped) > 0.5).astype(int)
            probabilities = model.predict(features_reshaped)
        elif name == 'Isolation Forest':
            prediction = model.predict(features_selected)
            prediction = np.where(prediction == 1, 0, 1)  # Invert predictions
            probabilities = model.score_samples(features_selected)
        elif name == 'OneDollar':
            prediction = np.array([model.recognize(x) for x in features_selected])
            probabilities = None
        else:
            prediction = model.predict(features_selected)
            probabilities = model.predict_proba(features_selected)

        results[name] = {
            'prediction': 'HUMAN' if prediction[0] == 1 else 'BOT',
            'probabilities': probabilities[0] if probabilities is not None else None
        }

    return results

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
    
    # Initialize and train models
    rf_model = build_random_forest()
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    lr_model = LogisticRegression(random_state=42, class_weight='balanced')
    lr_model.fit(X_train_resampled, y_train_resampled)
    
    if_model = build_isolation_forest()
    if_model.fit(X_train_resampled)
    
    lstm_model = build_lstm((1, X_train_resampled.shape[1]))
    lstm_model.fit(X_train_resampled.reshape((X_train_resampled.shape[0], 1, X_train_resampled.shape[1])), 
                   y_train_resampled, epochs=10, batch_size=32, verbose=0)
    
    od_model = OneDollarRecognizer()
    for x, y in zip(X_train_resampled, y_train_resampled):
        od_model.add_template(x, y)
    
    # Evaluate models
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model,
        'Isolation Forest': if_model,
        'LSTM': lstm_model,
        'OneDollar': od_model
    }
    
    for name, model in models.items():
        print(f"\n{name} Evaluation:")
        evaluate_model(model, X_test_selected, y_test, name)
    
    # Predict on test file
    test_file = "D:\\AbyssOrder\\Abyss-Order\\datasets2\\test2\\22227777.csv"
    results = predict_file(models, scaler, feature_selector, test_file, features.columns)
    
    for name, result in results.items():
        print(f"\n{name}: The file {test_file} is predicted to be from a {result['prediction']}.")
        if result['probabilities'] is not None:
            if name == 'Isolation Forest':
                print(f"Anomaly score: {result['probabilities'][0]:.4f}")
            else:
                print(f"Probabilities: Bot: {result['probabilities'][0]:.4f}, Human: {result['probabilities'][1]:.4f}")

if __name__ == "__main__":
    main()