import glob
import pandas as pd
from dollarpy import Recognizer, Template, Point
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)


# Preprocess data
def preprocess_data(df):
    df = df[['x_position', 'y_position']]  # Assuming 'x_position' and 'y_position' are the columns
    df.dropna(inplace=True)
    return df


# Train the $1 Recognizer model
def train_1dollar_model(human_files, bot_files):
    recognizer = Recognizer(templates=[])

    # Train on human data
    for file in human_files:
        df = preprocess_data(load_data(file))
        human_points = [Point(row['x_position'], row['y_position']) for _, row in df.iterrows()]
        recognizer.add_template(Template('Human', human_points))

    # Train on bot data
    for file in bot_files:
        df = preprocess_data(load_data(file))
        bot_points = [Point(row['x_position'], row['y_position']) for _, row in df.iterrows()]
        recognizer.add_template(Template('Bot', bot_points))

    return recognizer




# Function to visualize data distribution
def visualize_data_distribution(human_data, bot_data, test_data):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=human_data['x_position'], y=human_data['y_position'], cmap='Blues', shade=True, label='Human')
    sns.kdeplot(x=bot_data['x_position'], y=bot_data['y_position'], cmap='Reds', shade=True, label='Bot')
    sns.scatterplot(x=test_data['x_position'], y=test_data['y_position'], color='green', label='Test Data')
    plt.title('Mouse Movement Distribution')
    plt.legend()
    plt.show()


# Function to visualize movement speed
def visualize_movement_speed(human_data, bot_data, test_data):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(human_data['x_position'].diff().abs() + human_data['y_position'].diff().abs(), label='Human', color='blue')
    sns.kdeplot(bot_data['x_position'].diff().abs() + bot_data['y_position'].diff().abs(), label='Bot', color='red')
    sns.kdeplot(test_data['x_position'].diff().abs() + test_data['y_position'].diff().abs(), label='Test', color='green')
    plt.title('Movement Speed Distribution')
    plt.legend()
    plt.show()


# Function to visualize click patterns
def visualize_click_patterns(human_data, bot_data, test_data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=human_data['x_position'], y=human_data['y_position'], label='Human', color='blue')
    sns.scatterplot(x=bot_data['x_position'], y=bot_data['y_position'], label='Bot', color='red')
    sns.scatterplot(x=test_data['x_position'], y=test_data['y_position'], label='Test', color='green')
    plt.title('Click Patterns')
    plt.legend()
    plt.show()


# Safe scaling to avoid division by zero
def safe_scale(points):
    x_min = min(p.x for p in points)
    y_min = min(p.y for p in points)
    x_max = max(p.x for p in points)
    y_max = max(p.y for p in points)

    width = x_max - x_min
    height = y_max - y_min

    # Check if both width and height are zero
    if width == 0 and height == 0:
        return points  # Return points without scaling if there's no movement

    scale_factor = max(width, height)
    return [Point((p.x - x_min) / scale_factor, (p.y - y_min) / scale_factor) for p in points]


# Predict with the $1 recognizer
def predict_with_1dollar(recognizer, test_file):
    df = preprocess_data(load_data(test_file))
    points = [Point(row['x_position'], row['y_position']) for _, row in df.iterrows()]

    # Apply safe scaling before recognition
    points = safe_scale(points)

    # Proceed with recognition after scaling
    result = recognizer.recognize(points)
    return result


def main():
    # Load human and bot training data files
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
