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

# file = open("values.txt", "a")
randoms = []
stop = False
data = []
code = 0
date, year = [2222, 7777]
random.seed(date, year)
# file.write(str([date, year]))
# file.write("\n")
# file.close()

def on_move(x, y):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    data.append([now, x, y, None, None, None, None])

def on_click(x, y, button, pressed):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    data.append([now, x, y, button == mouse.Button.left, pressed, None, None])

def on_press(key):
    try:
        key.char
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        data.append([now, None, None, None, None, key.char, True])
    
    except:
        pass

def on_release(key):
    try:
        key.char
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        data.append([now, None, None, None, None, key.char, False])

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

def word():
    temp = ""
    characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  '.', ',', '?', '!', ':', ';', '"', "'", '-', '—', '(', ')', '[', ']', '{', '}', 
                  '+', '-', '*', '/', '=', '<', '>',
                  ]

    for i in range(random.randint(7, 15)):
        temp += characters[random.randint(0, 84)]

    return temp

def pack():
    global stop
    global code

    temp = int(code)
    code += 1

    if temp >= button_count:
        stop = True
        win.destroy()
        return

    buttons[temp].pack()
    buttons[temp].place(x=randoms[temp][0], y=randoms[temp][1])

    if len(randoms[temp]) == 3 and temp != 0:
        new = tk.Tk()
        new.geometry("500x200+0+0")
        label = tk.Label(new, text=randoms[temp][2])
        label.pack()
        text = tk.Text(new, height=5, width=20)
        text.pack()
        submit = tk.Button(new, text="submit", command=new.destroy)
        submit.pack()

    if temp != 0:
        buttons[temp - 1].destroy()

win = tk.Tk()
win.attributes('-fullscreen', True)
win.focus_force()

stop = False
file = open(f"{date}{year}.csv", "w", newline="\n")
writer = csv.writer(file)
writer.writerow(["timestamp", "x_position", "y_position", "button", "click", "key", "press"])
file.flush()
recent = datetime.datetime.now()

capture = Thread(target=capture_mouse)
keyboard = Thread(target=capture_keyboard)
periodic = Thread(target=periodic_save)

capture.start()
keyboard.start()
periodic.start()

a = win.winfo_screenwidth()
b = win.winfo_screenheight()
button_count = random.randint(10, 13)

for i in range(button_count):
    if i and random.randint(0, 1) and i != button_count - 1:
        randoms.append((random.randint(0, a), random.randint(0, b), word()))

    else:
        randoms.append((random.randint(0, a), random.randint(0, b)))

buttons = []
count = 1

for i in randoms:
    temp = tk.Button(win, text=count, command=pack)
    temp.pack_forget()
    
    buttons.append(temp)
    count += 1

pack()

win.mainloop()