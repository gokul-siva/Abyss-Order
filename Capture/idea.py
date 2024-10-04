from re import template
import tkinter as tk
import random

from click import command

randoms = []
code = 0

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
    global code

    temp = code
    code += 1

    buttons[temp].pack()
    buttons[temp].place(x=randoms[code][0], y=randoms[code][1])

win = tk.Tk()
win.attributes('-fullscreen', True)

a = win.winfo_screenwidth()
b = win.winfo_screenheight()

for i in range(random.randint(10, 13)):
    if random.randint(0, 1):
        randoms.append((random.randint(0, a), random.randint(0, b), word()))

    else:
        randoms.append((random.randint(0, a), random.randint(0, b)))

buttons = []
count = 1

for i in randoms:
    temp = tk.Button(win, text=count, command=lambda: (pack))
    temp.pack_forget()
    
    buttons.append(temp)
    count += 1

pack()

win.mainloop()