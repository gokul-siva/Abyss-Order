from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

# Path to your ChromeDriver executable

chrome_driver_path = "D:\AbyssOrder\Abyss-Order\ChromeDriver\chromedriver-win64\chromedriver-win64\chromedriver.exe"

# Check if the ChromeDriver file exists
if not os.path.isfile(chrome_driver_path):
    raise FileNotFoundError(f"ChromeDriver not found at {chrome_driver_path}")

# Set up the WebDriver service
service = Service(executable_path=chrome_driver_path)

# Initialize the WebDriver
driver = webdriver.Chrome(service=service)

# Access the CAPTCHA test site
#driver.get("https://www.google.com/recaptcha/api2/demo") 
driver.get("https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox.php")

try:
    # Wait for CAPTCHA to load and locate the checkbox
    wait = WebDriverWait(driver, 20)
    
    # Check if the CAPTCHA is inside an iframe and switch to it
    try:
        iframe = wait.until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        driver.switch_to.frame(iframe)
    except Exception as e:
        print("No iframe found or error switching to iframe:", e)
    
    # Use an alternative locator if necessary
    captcha_checkbox = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="recaptcha-anchor"]')))
    
    # Scroll the element into view
    driver.execute_script("arguments[0].scrollIntoView(true);", captcha_checkbox)
    
    # Optional wait to ensure the element is fully interactable
    time.sleep(1)
    
    # Use JavaScript click if standard click fails
    driver.execute_script("arguments[0].click();", captcha_checkbox)
    
    # Allow CAPTCHA validation time
    time.sleep(5)
finally:
    # Close the browser after automation
    driver.quit()
