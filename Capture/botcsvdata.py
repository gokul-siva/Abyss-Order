import pyautogui
import random
import string
import time
import datetime
import pandas as pd
import csv
import tkinter as tk
from pynput import mouse
from pynput.keyboard import Listener
from threading import Thread
import webbrowser

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

def sleep(n):
    for i in range(n):
        for j in range(1000):
            print("",end='')

def open_youtube_and_click():
    webbrowser.open("https://www.google.com/")
    #time.sleep(5)  # Wait for the browser to load
    
    pyautogui.moveTo(232, 68, duration=2)  # Move to search bar
    pyautogui.click()
    sleep(5000)
    pyautogui.typewrite("youtube", interval=0.1)
    pyautogui.press('enter')
    sleep(5000)
    
    pyautogui.moveTo(815, 467, duration=1.5)  # Move to a link
    pyautogui.click()
    sleep(7000)
    

    pyautogui.moveTo(723, 515, duration=1.5)  # Move to a link
    pyautogui.click()
    sleep(5000)

    
    pyautogui.moveTo(1620, 936, duration=1.5)  # Move to a link
    pyautogui.click()
    sleep(5000)

    
    
    pyautogui.moveTo(834, 120, duration=1.5)  # Move to a video
    pyautogui.click()
    sleep(10000)
    
    pyautogui.typewrite("genshin impact", interval=0.1)
    pyautogui.press('enter')
    sleep(5000)  # Watch video for 10 seconds

    pyautogui.scroll(-500)  # Scroll down on YouTube
    sleep(2000)

    pyautogui.moveTo(935, 995, duration=1.5)  # Move to a link
    pyautogui.click()
    sleep(5000)

    pyautogui.moveTo(1315, 615, duration=1.5)  # Move to a link
    pyautogui.click()
    sleep(5000)

    pyautogui.scroll(-900)  # Scroll down on YouTube
    sleep(2000)

    pyautogui.moveTo(1061, 725, duration=1.5)  # Move to a link
    pyautogui.click()
    sleep(5000)

def random_bot():
    #tasks = [open_youtube_and_click, search_random_word, browse_tabs]
    start_time = time.time()
    duration = 2 * 60  # Run for 3 minutes
    screen_width, screen_height = pyautogui.size()

    while time.time() - start_time < duration:
        action = random.choice(['move', 'click', 'scroll', 'drag', 'type'])
        x = random.randint(0, screen_width - 1)
        y = random.randint(0, screen_height - 1)

        if action == 'move':
            pyautogui.moveTo(x, y, duration=random.uniform(0.5, 2))
        
        elif action == 'click':
            pyautogui.click(x, y)
        
        elif action == 'scroll':
            pyautogui.scroll(random.randint(-100, 100), x, y)
        
        elif action == 'drag':
            x_end = random.randint(0, screen_width - 1)
            y_end = random.randint(0, screen_height - 1)
            pyautogui.dragTo(x_end, y_end, duration=random.uniform(0.5, 2))
        
        elif action == 'type':
            random_char = random.choice(string.ascii_letters)
            pyautogui.typewrite(random_char)

        time.sleep(random.uniform(0.5, 3))  # Random delay between actions
    #tasks=[open_youtube_and_click]
    #random.choice(tasks)() 

"""def random_bot():
    start_time = time.time()
    duration = 2* 60  # Run for 2 minutes
    screen_width, screen_height = pyautogui.size()

    while time.time() - start_time < duration:
        action = random.choice(['move', 'click', 'scroll', 'drag', 'type'])
        x = random.randint(0, screen_width - 1)
        y = random.randint(0, screen_height - 1)

        if action == 'move':
            pyautogui.moveTo(x, y, duration=random.uniform(0.5, 2))
        
        elif action == 'click':
            pyautogui.click(x, y)
        
        elif action == 'scroll':
            pyautogui.scroll(random.randint(-100, 100), x, y)
        
        elif action == 'drag':
            x_end = random.randint(0, screen_width - 1)
            y_end = random.randint(0, screen_height - 1)
            pyautogui.dragTo(x_end, y_end, duration=random.uniform(0.5, 2))
        
        elif action == 'type':
            random_char = random.choice(string.ascii_letters)
            pyautogui.typewrite(random_char)

        time.sleep(random.uniform(0.5, 3))  # Random delay between actions
"""


def start_random_bot():
    bot_thread = Thread(target=random_bot)
    bot_thread.start()

win = tk.Tk()
win.title("Thanks for your help")
win.geometry("500x300")

start = tk.Button(win, text="start capture", command=start_capture)
start.pack(pady=20)

upload = tk.Button(win, text="stop, upload", command=stop_process)
upload.pack(pady=10)

bot_button = tk.Button(win, text="Run Random Bot", command=start_random_bot)
bot_button.pack(pady=10)

win.mainloop()
