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
directory = 'D:\MouseDirectory'  # Updated path to D:/

# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)

# Global variables to control file numbering and tracking
person_counter_file = os.path.join(directory, 'counter.txt')
if not os.path.exists(person_counter_file):
    with open(person_counter_file, 'w') as f:
        f.write('1')

# Get the current person number and increment it for next use
def get_person_number():
    with open(person_counter_file, 'r+') as f:
        person_number = int(f.read().strip())
        if person_number > 10:
            print("Data collection complete. 10 files created.")
            sys.exit()  # Exit the program after creating 10 files
        f.seek(0)
        f.write(str(person_number + 1))
    return person_number

# Function to generate filename for the current run
def get_filename():
    person_number = get_person_number()
    return os.path.join(directory, f'Person{person_number}.csv')

# List to store mouse movements
mouse_data = []
stop_tracking = False  # Flag to stop tracking after 2 minutes

# Mouse movement tracking function
def on_move(x, y):
    if not stop_tracking:  # Only track if the flag is False
        # Record timestamp, X and Y coordinates
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mouse_data.append([timestamp, x, y])

# Set the sampling rate (50ms = 0.05 seconds)
sampling_interval = 0.05  # 50 ms

def run_mouse_tracker():
    # Start mouse listener
    with mouse.Listener(on_move=on_move) as listener:
        while not stop_tracking:  # Continuously run in the background unless stopped
            time.sleep(sampling_interval)
        listener.stop()  # Stop the mouse listener after 2 minutes

# Background collection and saving
def save_mouse_data(filename):
    while not stop_tracking:
        # Save the collected data to CSV every 30 seconds
        time.sleep(30)  # Adjust as needed
        if mouse_data:
            df = pd.DataFrame(mouse_data, columns=['Timestamp', 'X', 'Y'])
            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
            mouse_data.clear()  # Clear the buffer after saving

# Function to stop the program after 2 minutes
def stop_after_duration(duration=120):  # Duration in seconds (120s = 2 minutes)
    global stop_tracking
    time.sleep(duration)
    stop_tracking = True
    print("2 minutes passed, stopping data collection...")

# Function to add script to Windows startup
def add_to_startup():
    startup_folder = os.path.join(os.getenv('APPDATA'), 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
    script_path = os.path.realpath(sys.argv[0])
    
    # Copy the script to the startup folder
    shutil.copy(script_path, os.path.join(startup_folder, 'mouse_tracker.py'))

# Show a message box to get permission
def show_permission_dialog():
    return ctypes.windll.user32.MessageBoxW(0, 
            "Do you allow this program to automatically collect your mouse movements and start on Windows boot?", 
            "Permission Required", 1)

# Main function to run mouse tracking and add to startup if permission granted
def main():
    # Ask for user permission
    response = show_permission_dialog()
    
    if response == 1:  # If 'Yes' is selected
        # Add the script to startup
        add_to_startup()
        print("Program added to startup and data collection will begin.")

        for _ in range(10):  # Run 10 times for Person1.csv to Person10.csv
            # Generate a unique filename for each 2-minute run
            filename = get_filename()
            print(f"Saving data to {filename}")

            # Start mouse tracking
            tracker_thread = Thread(target=run_mouse_tracker)
            tracker_thread.daemon = True
            tracker_thread.start()

            # Start saving data in the background
            saving_thread = Thread(target=save_mouse_data, args=(filename,))
            saving_thread.daemon = True
            saving_thread.start()

            # Start the timer to stop the program after 2 minutes
            stop_thread = Thread(target=stop_after_duration, args=(120,))  # 120 seconds = 2 minutes
            stop_thread.daemon = True
            stop_thread.start()

            # Wait for the threads to finish
            stop_thread.join()
            tracker_thread.join()
            saving_thread.join()

    else:
        print("Permission not granted. Program will not run in the background.")

if __name__ == "__main__":
    main()
