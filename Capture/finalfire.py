from pynput import mouse
from pynput.keyboard import Listener
from threading import Thread
import tkinter as tk
import time
import datetime
import pandas as pd
import csv
import math
import firebase_admin
from firebase_admin import credentials, db



##use \\ in the path
path_to_json="C:\\Users\\sripr\\Downloads\\mouse2772-9e216-firebase-adminsdk-o884w-97a8cd3c97.json" 



stop = False
data = []
data_frame = pd.DataFrame()

def on_move(x, y):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    data.append([now, x, y, None, None, 0, 0, None, None])

def on_scroll(x, y, dx, dy):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    data.append([now, x, y, None, None, dx, dy, None, None])

def on_click(x, y, button, pressed):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    data.append([now, x, y, button == mouse.Button.left, pressed, 0, 0, None, None])

def on_press(key):
    try:
        key.char
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        data.append([now, None, None, None, None, 0, 0, key.char, True])
    
    except:
        pass

def on_release(key):
    try:
        key.char
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        data.append([now, None, None, None, None, 0, 0, key.char, False])

    except:
        pass

def capture_mouse():
    with mouse.Listener(on_move=on_move, on_scroll=on_scroll, on_click=on_click) as listen:
        while not stop:
            time.sleep(0.5)
        listen.stop()

def capture_keyboard():
    with Listener(on_press=on_press, on_release=on_release) as listen:
        while not stop:
            time.sleep(0.5)
        listen.stop()

def periodic_save():
    global recent
    global data
    now = datetime.datetime.now()

    while not stop:
        while (now - recent).total_seconds() <= 150:
            time.sleep(50)
            now = datetime.datetime.now()

        try:
            data[0]
            writer.writerows(data)
            data = []
            file.flush()

        except:
            pass

        recent = datetime.datetime.now()

def stop_process():
    global stop
    global data

    stop = True
    writer.writerows(data)
    data = []
    file.flush()
    file.close()

    # Upload data to Firebase
    upload_to_firebase("send.csv")

def start_capture():
    global capture
    global keyboard
    global stop
    global file
    global writer
    global recent

    stop = False
    file = open("send.csv", "w", newline="\n")
    writer = csv.writer(file)
    writer.writerow(["timestamp", "x_position", "y_position", "button", "click", "dx", "dy", "key", "press"])
    file.flush()
    recent = datetime.datetime.now()
    
    capture = Thread(target=capture_mouse)
    keyboard = Thread(target=capture_keyboard)
    periodic = Thread(target=periodic_save)

    capture.start()
    keyboard.start()
    periodic.start()

def is_json_compliant(value):
    # Check if the value is a valid JSON number
    if isinstance(value, float):
        return not (math.isnan(value) or math.isinf(value))
    return True

def sanitize_value(value):
    try:
        float_val = float(value)
        if is_json_compliant(float_val):
            return float_val
        else:
            return 0.0  # Replace invalid float values with 0.0
    except ValueError:
        return value  # If not a float, return as is
    

def upload_to_firebase(csv_file_path):
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(path_to_json)  # Replace with your service account key file path
    firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mouse2772-9e216-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase database URL
})
    
    # Load the CSV data
    df = pd.read_csv(csv_file_path)

    # Reference the Firebase DB using "name_of_user"
    ref = db.reference(name_of_user)
    stamp=0
    # Upload each row of data under the corresponding timestamp
    for _, row in df.iterrows():
        sanitized_row = {key: sanitize_value(value) for key, value in row.items()}
        stamp=stamp+1
        ref.child(str(stamp)).set(sanitized_row)

    print(f"Data from {csv_file_path} uploaded successfully.")

    # Show the termination button
    show_termination_button()

def terminate_program():
    print("Program terminated.")
    exit(0)

# Tkinter GUI
def start_gui():
    global name_of_user

    def start_app():
        global name_of_user
        name_of_user = entry.get()
        start_capture()

    win = tk.Tk()
    win.title("Thanks for your help")
    win.geometry("500x300")

    label = tk.Label(win, text="Enter your name:")
    label.pack(pady=10)

    entry = tk.Entry(win)
    entry.pack(pady=10)

    start = tk.Button(win, text="Start Capture", command=start_app)
    start.pack(pady=20)

    upload = tk.Button(win, text="Stop and Upload", command=stop_process)
    upload.pack(pady=10)

    win.mainloop()

def show_termination_button():
    win = tk.Tk()
    win.title("Terminate Program")
    win.geometry("300x150")

    terminate = tk.Button(win, text="Terminate Program", command=terminate_program)
    terminate.pack(pady=20)

    win.mainloop()

if __name__ == "__main__":
    start_gui()
