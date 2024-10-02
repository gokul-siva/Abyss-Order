import pyautogui
import random
import time
import string

def random_bot():
    start_time = time.time()
    duration = 3 * 60  # Run for 3 minutes
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
random_bot()
