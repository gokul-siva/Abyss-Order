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
import pyautogui

randoms = []
stop = False
data = []
code = 0
date, year = [2222, 7777]
random.seed(date, year)

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

a = 1536
b = 864
button_count = random.randint(10, 13)

for i in range(button_count):
    if i and random.randint(0, 1) and i != button_count - 1:
        randoms.append((random.randint(0, a) * 1.25 + 9, random.randint(0, b) * 1.25 + 9, word()))

    else:
        randoms.append((random.randint(0, a) * 1.25 + 9, random.randint(0, b) * 1.25 + 9))
print(randoms)
time.sleep(10)

for i in randoms:
    if len(i) == 3:
        pyautogui.moveTo(338, 144, duration=random.random()*3)
        pyautogui.click()
        pyautogui.typewrite(i[2])
        pyautogui.moveTo(338, 191, duration=random.random()*3)
        pyautogui.click()
    pyautogui.moveTo(i[0], i[1], duration=random.random()*3)
    pyautogui.click