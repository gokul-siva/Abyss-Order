from django.shortcuts import render
import json
from datetime import datetime as dt
import csv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import os
from django.shortcuts import redirect
import csv
import time

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
    encoder = OneHotEncoder(categories=categories, sparse=False)
    encoded = encoder.fit_transform(data[["key"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['key']))
    df_encoded = pd.concat([data.drop('key', axis=1), encoded_df], axis=1)
    df_encoded = df_encoded.drop("key_ae", axis=1)
    df_encoded["button"] = df_encoded["button"].astype(float)
    df_encoded["click"] = df_encoded["click"].astype(float)
    df_encoded["press"] = df_encoded["press"].astype(float)
    
    sequence = create_sequences(df_encoded, 63, 58)
    
    sequence[np.isnan(sequence)] = 0
    
    
    sequence = torch.from_numpy(sequence)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    
   

    return sequence

def predict(sequence):
    model = LSTMClassifier(input_size=94, hidden_layer_size=188, output_size=2)
    model.load_state_dict(torch.load("D:\AbyssOrder\Abyss-Order\model.pth"))
    model.eval()

    with torch.no_grad():
        
        prediction = model(sequence)

    prediction_class = torch.argmax(prediction, dim=1)
    count = 0

    for i in prediction_class:
        if i == 1:
            count += 1
    return count / len(prediction_class)
        


def index(request):
    return render(request, 'bot_detection/index.html')


@csrf_exempt
def upload_data(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        with open('data.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file,fieldnames=['timestamp','x_position','y_position','button','click','dx','dy','key','press'])
            for event in data:
                if event['type'] == 'mouse':
                    writer.writerow({'timestamp':event['timestamp'],'x_position':event['x_position'],'y_position':event['y_position'],\
                                     'button':'','click':'','dx':0,'dy':0,'key':'','press':''})
                    #writer.writerow([event['timestamp'], event['x_position'], event['y_position'], 'MouseMove'])
                elif event['type'] == 'click':
                    if event['button'] =="left":
                        writer.writerow({'timestamp':event['timestamp'],'x_position':event['x_position'],'y_position':event['y_position'],\
                                     'button':True,'click':True,'dx':0,'dy':0,'key':'','press':''})
                    else:
                        writer.writerow({'timestamp':event['timestamp'],'x_position':event['x_position'],'y_position':event['y_position'],\
                                     'button':False,'click':True,'dx':0,'dy':0,'key':'','press':''})
                
                elif event['type'] == 'keyboard':
                    if event['press']== 'true':
                    #writer.writerow([event['timestamp'], event['key'], 'KeyStroke'])
                        writer.writerow({'timestamp':event['timestamp'],'x_position':'','y_position':'',\
                                     'button':'','click':'','dx':0,'dy':0,'key':event['key'],'press':True})
                    else:
                        writer.writerow({'timestamp':event['timestamp'],'x_position':'','y_position':'',\
                                     'button':'','click':'','dx':0,'dy':0,'key':event['key'],'press':False})

        return JsonResponse({'status': 'success'})
    



def capture_data_view(request):
    if request.method == "POST":
       
        data = request.POST.getlist('data[]')

        
        filename = f'data_capture_0.csv'
        file_path = os.path.join('data_storage', filename)
        

        
        with open(file_path, 'w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['timestamp','x_position','y_position','button','click','dx','dy','key','press'])
            writer.writeheader()
            writer = csv.writer(f)
            for row in data:
                row=eval(row)
               
                #print("Each row:\n",row)
                for rowdict in row:
                   
                    timestamp=rowdict["timestamp"]  
                    timestamp=timestamp.replace("T"," ")
                    timestamp=timestamp.replace("Z"," ")
                   
                    
                    if "x_position" in rowdict.keys():
                        xpos=rowdict["x_position"]
                    else:
                        xpos=''
                    if "y_position" in rowdict.keys():
                        ypos=rowdict["y_position"]
                    else:
                        ypos=''
                    if "key" in rowdict.keys():
                        key=rowdict["key"]
                        if key not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                    '.', ',', '?', '!', ':', ';', '"', "'", '-', '(', ')', '[', ']', '{', '}', 
                    '+', '*', '/', '=', '<', '>', "ae", "#", "_", "|"
                    ]:
                            continue
                        
                    else:
                        key=''
                    if "button" in rowdict.keys():
                        if rowdict['button']=='left':
                            button=True
                        else:
                            button=False
                    else:
                        button=''
                    if "click" in rowdict.keys():
                        click=rowdict['click']
                    else:
                        click=''
                    dx=0
                    dy=0
                    if "press" in rowdict.keys():
                        if rowdict['press']=='true':
                            press=True
                        else:
                            press=False
                    else:
                        press=''
                    writer.writerow([timestamp,xpos,ypos,button,click,dx,dy,key,press])
                    

        sequence = preprocess(file_path)
        bot_prediction = predict(sequence)
        print("Bot predicted: ",bot_prediction)
        print("Bot Threshhold: 0.8")
        
        if bot_prediction > 0.8:
            return JsonResponse({'captcha': True}) 
        else:
            return JsonResponse({'captcha': False})

    return render(request, 'index.html')


def display(request):
    date=dt.now()
    msg="Hello!"
    t_dict={'DATE':date,'str':msg}
    return render(request,'index.html',context=t_dict)

