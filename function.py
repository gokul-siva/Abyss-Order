import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length, overlap):
    sequences = []
    result = []
    count = 0
    
    for i in range(0, len(data) - seq_length + 1, seq_length - overlap):
        count += 1
        sequence = data.iloc[i:i + seq_length]
        sequence = sequence.values
        sequences.append(sequence)

    return np.array(sequences)

def preprocess(file):
    data = pd.read_csv(file)
    data = data.fillna(-1)
    data = data.replace(True, 1)
    data = data.replace(False, 0)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["timestamp"] = data["timestamp"].diff().dt.total_seconds()
    data["speed"] = np.sqrt(data["x_position"].diff() ** 2 + data["y_position"].diff() ** 2) / data["timestamp"]
    data["moved"] = data["x_position"].diff() + data["y_position"].diff()
    data = data.fillna(method="ffill").iloc[1:, :]
    data = data.fillna(0)
    data.reset_index(drop=True, inplace=True)

    categories = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                    '.', ',', '?', '!', ':', ';', '"', "'", '-', '(', ')', '[', ']', '{', '}', 
                    '+', '*', '/', '=', '<', '>', "ae", "#", "_", "|"
                    ]]

    data["key"] = data["key"].replace(-1, "ae")
    encoder = OneHotEncoder(categories=categories, sparse_output=False)
    encoded = encoder.fit_transform(data[["key"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['key']))
    df_encoded = pd.concat([data.drop('key', axis=1), encoded_df], axis=1)
    df_encoded = df_encoded.drop("key_ae", axis=1)
    df_encoded["button"] = df_encoded["button"].astype(float)
    df_encoded["click"] = df_encoded["click"].astype(float)
    df_encoded["press"] = df_encoded["press"].astype(float)

    sequence = create_sequences(df_encoded, 50, 30)
    sequence[np.isnan(sequence)] = 0

    sequence = torch.from_numpy(sequence)
    sequence = torch.tensor(sequence, dtype=torch.float32)

    return sequence

def predict(sequence):
    model = LSTMClassifier(input_size=94, hidden_layer_size=188, output_size=2)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    with torch.no_grad():
        prediction = model(sequence)

    prediction_class = torch.argmax(prediction, dim=1)
    count = 0

    for i in prediction_class:
        if i == 1:
            count += 1
    print("Bot percentage:", count / len(prediction_class))

predict(preprocess("E:\\Abyss-Order\\datasets2\\test\\keshav.csv"))