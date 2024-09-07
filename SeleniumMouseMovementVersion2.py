from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
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

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--start-maximized")

# Set up the WebDriver service
service = Service(executable_path=chrome_driver_path)

# Initialize the WebDriver
driver = webdriver.Chrome(service=service, options=chrome_options)
print("WebDriver initialized")

# Access the CAPTCHA test site
driver.get("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox.php")
print("Accessed CAPTCHA test site")

try:
    # Wait for CAPTCHA to load and locate the checkbox
    wait = WebDriverWait(driver, 20)
    
    # Find all iframes on the page
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    print(f"Found {len(iframes)} iframes")

    # Switch to the reCAPTCHA iframe
    recaptcha_iframe = next((iframe for iframe in iframes if "recaptcha" in iframe.get_attribute("src")), None)
    if recaptcha_iframe:
        driver.switch_to.frame(recaptcha_iframe)
        print("Switched to reCAPTCHA iframe")
    else:
        print("reCAPTCHA iframe not found")

    # Locate the CAPTCHA checkbox
    captcha_checkbox = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "recaptcha-checkbox-border")))
    print("CAPTCHA checkbox located")

    # Get the location and size of the checkbox
    location = captcha_checkbox.location_once_scrolled_into_view
    size = captcha_checkbox.size
    
    # Calculate the center of the checkbox
    x = location['x'] + size['width'] / 2
    y = location['y'] + size['height'] / 2
    
    # Get the position of the browser window
    browser_x = driver.execute_script("return window.screenX;")
    browser_y = driver.execute_script("return window.screenY;")
    
    # Calculate the absolute screen coordinates
    screen_x = browser_x + x
    screen_y = browser_y + y + driver.execute_script("return window.outerHeight - window.innerHeight;")
    
    print(f"Calculated CAPTCHA checkbox center coordinates: x={screen_x}, y={screen_y}")
    
    # Allow time to switch focus to the browser window
    print("Waiting for 5 seconds. Please switch focus to the browser window.")
    time.sleep(5)
    
    # Debug: Get current mouse position
    current_x, current_y = pyautogui.position()
    print(f"Current mouse position: x={current_x}, y={current_y}")
    
    # Move mouse to the checkbox using PyAutoGUI
    print("Moving mouse to CAPTCHA checkbox")
    pyautogui.moveTo(int(screen_x), int(screen_y), duration=1)
    
    # Debug: Get new mouse position
    new_x, new_y = pyautogui.position()
    print(f"New mouse position: x={new_x}, y={new_y}")
    
    # Click the checkbox using PyAutoGUI
    print("Clicking CAPTCHA checkbox")
    pyautogui.click()
    
    # Allow CAPTCHA validation time
    print("Waiting for CAPTCHA validation")
    time.sleep(5)
    
    # Check if the CAPTCHA was successfully clicked
    try:
        driver.switch_to.default_content()  # Switch back to the main content
        success_message = driver.find_element(By.CLASS_NAME, "recaptcha-success")
        print("CAPTCHA successfully solved!")
    except:
        print("CAPTCHA may not have been successfully solved. Please check manually.")
    
    print("CAPTCHA interaction complete")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print("Full traceback:")
    print(traceback.format_exc())

finally:
    # Capture a screenshot for debugging
    screenshot_path = "captcha_debug_screenshot.png"
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved as {screenshot_path}")

    # Close the browser after automation
    driver.quit()
    print("Browser closed")

print("Script ended")