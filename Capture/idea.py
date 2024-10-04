from re import template
import tkinter as tk
import random

file = open("values.txt", "a")
randoms = []
code = 0
date = 904
year = 2005
random.seed(date, year)
file.write(str([date, year]))
file.write("\n")
file.close()

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

    if temp == button_count - 1:
        print(temp, button_count)
        win.destroy()
        return

    buttons[temp].pack()
    buttons[temp].place(x=randoms[code][0], y=randoms[code][1])

    if len(randoms[temp]) == 3 and temp != 0:
        new = tk.Tk()
        label = tk.Label(new, text=randoms[temp][2])
        label.pack()
        text = tk.Text(new, height=5, width=20)
        text.pack()
        submit = tk.Button(new, text="submit", command=lambda: new.destroy() if text.get(1.0, "end-1c") == randoms[temp][2] else print(text.get(1.0, "end-1c")))
        submit.pack()

    if temp != 0:
        buttons[temp - 1].destroy()
    
win = tk.Tk()
win.attributes('-fullscreen', True)
win.focus_force()

a = win.winfo_screenwidth()
b = win.winfo_screenheight()
button_count = random.randint(10, 13)

for i in range(button_count):
    if i and random.randint(0, 1) and i != button_count - 1:
        randoms.append((random.randint(0, a), random.randint(0, b), word()))

    else:
        randoms.append((random.randint(0, a), random.randint(0, b)))

print(randoms)
buttons = []
count = 1

for i in randoms:
    temp = tk.Button(win, text=count, command=pack)
    temp.pack_forget()
    
    buttons.append(temp)
    count += 1

pack()

win.mainloop()