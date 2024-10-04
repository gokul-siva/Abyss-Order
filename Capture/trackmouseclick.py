import pyautogui
import tkinter as tk
from pynput import mouse
import threading


def on_click(x, y, button, pressed):
    if pressed:
        # Print the coordinates when the mouse is clicked
        print(f"Mouse clicked at ({x}, {y})")

def start_mouse_listener():
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()

# GUI setup to start the listener
def run_listener():
    listener_thread = threading.Thread(target=start_mouse_listener)
    listener_thread.start()

# Tkinter GUI to start the coordinate tracker
win = tk.Tk()
win.title("Mouse Coordinate Tracker")
win.geometry("300x200")

start_button = tk.Button(win, text="Start Tracking", command=run_listener)
start_button.pack(pady=50)

win.mainloop()
