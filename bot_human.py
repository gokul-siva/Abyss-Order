from re import template
import tkinter as tk
import random
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

def on_move(x, y):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    data.append([now, x, y, None, None, 0, 0, None, None])

def on_click(x, y, button, pressed):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    data.append([now, x, y, button == mouse.Button.left, pressed, 0, 0, None, None])

def on_press(key):
    try:
        key.char
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        data.append([now, None, None, None, None, 0, 0, key.char, True])
    
    except:
        pass

def on_release(key):
    try:
        key.char
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        data.append([now, None, None, None, None, 0, 0, key.char, False])

    except:
        pass

def capture_mouse():
    with mouse.Listener(on_move=on_move, on_click=on_click) as listen:
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
        while (now - recent).total_seconds() <= 5:
            time.sleep(5)
            now = datetime.datetime.now()

        try:
            data[0]
            writer.writerows(data)
            data = []
            file.flush()

        except:
            pass

        recent = datetime.datetime.now()