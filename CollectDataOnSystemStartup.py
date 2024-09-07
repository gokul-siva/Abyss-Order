import os
import pandas as pd
import ctypes
import time
from pynput import mouse
from datetime import datetime
from threading import Thread
import shutil
import sys

# Directory to save the mouse data
directory = 'D:\AbyssOrder\Abyss-Order\MouseDirectory'

# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

# Global variables to control file numbering and tracking
person_counter_file = os.path.join(directory, 'counter.txt')
if not os.path.exists(person_counter_file):
    with open(person_counter_file, 'w') as f:
        f.write('1')

def get_person_number():
    with open(person_counter_file, 'r+') as f:
        person_number = int(f.read().strip())
        if person_number > 10:
            print("Data collection complete. 10 files created.")
            sys.exit()  # Exit the program after creating 10 files
        f.seek(0)
        f.write(str(person_number + 1))
    return person_number

def get_filename():
    person_number = get_person_number()
    return os.path.join(directory, f'Person{person_number}.csv')

mouse_data = []
stop_tracking = False

def on_move(x, y):
    if not stop_tracking:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mouse_data.append([timestamp, x, y])

sampling_interval = 0.05

def run_mouse_tracker():
    with mouse.Listener(on_move=on_move) as listener:
        while not stop_tracking:
            time.sleep(sampling_interval)
        listener.stop()

def save_mouse_data(filename):
    while not stop_tracking or mouse_data:
        if mouse_data:
            df = pd.DataFrame(mouse_data, columns=['Timestamp', 'X', 'Y'])
            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
            mouse_data.clear()
        time.sleep(10)  # Save every 10 seconds to ensure data is not lost

def stop_after_duration(duration=120):
    global stop_tracking
    time.sleep(duration)
    stop_tracking = True
    print("2 minutes passed, stopping data collection...")

def add_to_startup():
    startup_folder = os.path.join(os.getenv('APPDATA'), 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
    script_path = os.path.realpath(sys.argv[0])
    shutil.copy(script_path, os.path.join(startup_folder, 'mouse_tracker.py'))

def show_permission_dialog():
    return ctypes.windll.user32.MessageBoxW(0, 
            "Do you allow this program to automatically collect your mouse movements and start on Windows boot?", 
            "Permission Required", 1)

def main():
    response = show_permission_dialog()
    
    if response == 1:
        add_to_startup()
        print("Program added to startup and data collection will begin.")

        for _ in range(10):
            filename = get_filename()
            print(f"Saving data to {filename}")

            global mouse_data, stop_tracking
            mouse_data = []
            stop_tracking = False

            tracker_thread = Thread(target=run_mouse_tracker)
            tracker_thread.daemon = True
            tracker_thread.start()

            saving_thread = Thread(target=save_mouse_data, args=(filename,))
            saving_thread.daemon = True
            saving_thread.start()

            stop_thread = Thread(target=stop_after_duration, args=(120,))
            stop_thread.daemon = True
            stop_thread.start()

            # Check for file existence every 10 seconds
            while not stop_tracking:
                time.sleep(10)
                if os.path.exists(filename):
                    print(f"File {filename} exists in the directory.")
                else:
                    print(f"File {filename} does not exist yet.")

            stop_thread.join()
            tracker_thread.join()
            saving_thread.join()

            print(f"Data collection for {filename} complete.")

    else:
        print("Permission not granted. Program will not run in the background.")

if __name__ == "__main__":
    main()
