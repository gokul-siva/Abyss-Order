from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import pyautogui

print("Script started")

# Path to your ChromeDriver executable
chrome_driver_path = "D:\AbyssOrder\Abyss-Order\ChromeDriver\chromedriver-win64\chromedriver-win64\chromedriver.exe"

# Check if the ChromeDriver file exists
if not os.path.isfile(chrome_driver_path):
    raise FileNotFoundError(f"ChromeDriver not found at {chrome_driver_path}")

print(f"ChromeDriver path verified: {chrome_driver_path}")

# Set up the WebDriver service
service = Service(executable_path=chrome_driver_path)

# Initialize the WebDriver
driver = webdriver.Chrome(service=service)
print("WebDriver initialized")

# Access the CAPTCHA test site
driver.get("https://www.google.com/recaptcha/api2/demo")
print("Accessed CAPTCHA test site")

try:
    # Wait for CAPTCHA to load and locate the checkbox
    wait = WebDriverWait(driver, 20)
    
    # Check if the CAPTCHA is inside an iframe and switch to it
    try:
        iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        driver.switch_to.frame(iframe)
        print("Switched to iframe")
    except Exception as e:
        print("No iframe found or error switching to iframe:", e)
    
    # Use an alternative locator if necessary
    captcha_checkbox = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="recaptcha-anchor"]')))
    print("CAPTCHA checkbox located")
    
    # Scroll the element into view
    driver.execute_script("arguments[0].scrollIntoView(true);", captcha_checkbox)
    print("Scrolled to CAPTCHA checkbox")
    
    # Get the location and size of the checkbox
    location = captcha_checkbox.location
    size = captcha_checkbox.size
    
    # Calculate the center of the checkbox
    x = location['x'] + size['width'] / 2
    y = location['y'] + size['height'] / 2
    
    # Convert browser coordinates to screen coordinates
    screen_x = x + driver.execute_script("return window.screenX;")
    screen_y = y + driver.execute_script("return window.screenY;") + driver.execute_script("return window.outerHeight - window.innerHeight;")
    
    print(f"CAPTCHA checkbox center coordinates: x={screen_x}, y={screen_y}")
    
    # Allow time to switch focus to the browser window
    print("Waiting for 5 seconds. Please switch focus to the browser window.")
    time.sleep(5)
    
    # Move mouse to the checkbox using PyAutoGUI
    print("Moving mouse to CAPTCHA checkbox")
    pyautogui.moveTo(int(screen_x), int(screen_y), duration=1)
    
    # Click the checkbox using PyAutoGUI
    print("Clicking CAPTCHA checkbox")
    pyautogui.click()
    
    # Allow CAPTCHA validation time
    print("Waiting for CAPTCHA validation")
    time.sleep(5)
    
    print("CAPTCHA interaction complete")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print("Full traceback:")
    print(traceback.format_exc())

finally:
    # Close the browser after automation
    driver.quit()
    print("Browser closed")

print("Script ended")