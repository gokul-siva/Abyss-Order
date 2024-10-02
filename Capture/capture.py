from pynput import mouse
from pynput.keyboard import Listener
from threading import Thread
import tkinter as tk
import time
import datetime
import pandas as pd
import csv

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
            print("Inside", (now - recent).total_seconds())
            time.sleep(50)
            now = datetime.datetime.now()

        try:
            data[0]
            writer.writerows(data)
            data = []
            file.flush()

        except:
            print("Indside except", data)
            pass

        recent = datetime.datetime.now()
        print((now - recent).total_seconds())

def stop_process():
    global stop
    stop = True
    writer.writerows(data)
    data = []

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

win = tk.Tk()
win.title("Thanks for your help")
win.geometry("500x300")

start = tk.Button(win, text="start capture", command=start_capture)
start.pack(pady=20)

upload = tk.Button(win, text="stop, upload", command=stop_process)
upload.pack(pady=10)
win.mainloop()