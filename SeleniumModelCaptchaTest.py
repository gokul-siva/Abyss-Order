'''import numpy as np
import time
from selenium.webdriver.common.action_chains import ActionChains

# Simulate human-like mouse movements using the trained model
def simulate_mouse_movements(driver, model, start_point, end_point, steps=100):
    actions = ActionChains(driver)
    
    current_position = np.array(start_point)
    for _ in range(steps):
        # Predict next movement (simulate a small time step)
        movement = model.predict([current_position])[0]
        next_position = np.clip(current_position + movement, [0, 0], end_point)
        
        # Move the mouse in small increments
        actions.move_by_offset(next_position[0] - current_position[0],
                               next_position[1] - current_position[1]).perform()
        
        current_position = next_position
        time.sleep(0.05)  # Small delay to make movement appear natural

# Start the Selenium browser
driver = webdriver.Chrome(executable_path="D:\ChromeDriver\chromedriver-win64\chromedriver-win64\chromedriver.exe")
driver.get("https://www.google.com/recaptcha/api2/demo")

# Wait for CAPTCHA to load
time.sleep(3)

# Locate CAPTCHA checkbox
captcha_checkbox = driver.find_element(By.ID, "recaptcha-anchor")

# Get the start position of the mouse
start_pos = (0, 0)
end_pos = captcha_checkbox.location['x'], captcha_checkbox.location['y']

# Simulate mouse movement towards the CAPTCHA checkbox
simulate_mouse_movements(driver, model, start_pos, end_pos)

# Click the checkbox once reached
captcha_checkbox.click()

# Close browser
time.sleep(5)
driver.quit()
'''