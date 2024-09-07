import pyautogui
import time

# Wait for 5 seconds to switch to the target window
time.sleep(5)

# Move the mouse to position (500, 500) on the screen
pyautogui.moveTo(500, 500, duration=1)

# Click at the current mouse position
pyautogui.click()
