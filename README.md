**INTEL GEN AI HACKATHON 2024- KPR COLLEGE OF ENGINEERING,COIMBATORE**

**Team: Abyss Order**

Members: Gokulakannan, Sri Prasad

**Problem statement:**

To implement a passive captcha.
Normal captcha requires you to type distored text or select a check box. This repository implements a LSTM to analyse the user behaviour from the sequence of mouse cursor positions, keystrokes, mouse clicks and classify the user as a potential bot. 


**Our AI Solution**

This project presents a passive CAPTCHA using AI that detects bot-like behavior without requiring user interaction. By analyzing a user's activity, such as mouse movements, clicks, and keystrokes, within a short window of just 10 seconds, we can predict if the user is human or bot.

The backbone of this system is a Long Short-Term Memory (LSTM) model trained to classify user behaviors based on input sequences. The model leverages behavioral data like mouse positions, clicks, and keyboard presses to differentiate between humans and bots with high accuracy.


**With this solution:**

1)No more distorted texts.
2)No more image selection puzzles.
3)No more frustrating captchas.
All that’s required to prove you're human is to use the website normally!


**How It Works**

Data Capturing
Our AI model gathers data from:

1)Mouse Movements (x and y positions)
2)Mouse Clicks (button type and click status)
3)Keystrokes (key pressed and whether it's a hold or release)
The activity is captured automatically during interaction with the website and stored in CSV format. This data is then preprocessed to create sequences that are fed into the LSTM model.


**Model Architecture**

The core of the solution is an LSTM neural network that processes sequences of user actions and classifies them as either human or bot behavior. Here's how it’s structured:

Input Layer: Takes the sequence of actions (mouse, keyboard, etc.).
LSTM Layer: Extracts temporal patterns from the data.
Fully Connected Layer: Maps the LSTM outputs to a classification result—either human or bot.


**Model Training**

The model was trained using real-world data from both human users and bots. Data was collected through custom-built GUIs, capturing the natural interaction patterns of humans versus programmed bot behavior. The model was trained in two phases using Jupyter notebooks:

Training.ipynb: For training the LSTM model.
Testing.ipynb: For evaluating the model’s accuracy and generalization.

**Data Collection**

Datasets were collected from various individuals and bot simulations:

datasets/ and datasets2/: Contain CSV files of captured user activity.
Capture/: Contains the Python scripts (Capture_gui.py and bot_gui.py) used to collect data through a GUI interface using Tkinter.



**Project Structure**

Here is a brief overview of the folder structure:

BotDetectionProject/
├── datasets/                # Collected human data
├── datasets2/               # Additional datasets (bots)
├── templates/
│   ├── bot_detection/
│   │   ├── index.html       # Main frontend template
├── static/
│   ├── css/                 # Styling for the website
├── training.ipynb           # Model training notebook
├── testing.ipynb            # Model testing notebook
└── manage.py                # Django management script
├── Capture/
│   ├── Capture_gui.py       # For capturing user data through GUI
│   ├── bot_gui.py           # Simulated bot data collection
.   .
.

**Instructions to Run the Application to Check Whwther you are a human or not (Just 3 steps!)**
To run the passive CAPTCHA Django application, follow these steps:

1)Install the Required Dependencies:
      pip install django torch pandas numpy scikit-learn

2)Navigate to the Project Directory:
  Open your command prompt or terminal and navigate to the BotDetectionProject folder:
      cd BotDetectionProject

3)Run the Django Application:
  Start the Django server by running the following command:
      python manage.py runserver


**Access the Application**

Open your browser and go to http://127.0.0.1:8000/ to view the main page of the bot detection system.

**Future Scope:**

Model Enhancement: Improving the LSTM model by integrating additional behavioral features.
Scalability: Making the solution scalable to support high-traffic websites.
Generalization: Increasing the dataset to train the model with various behaviors and bots.
Real-Time Detection: Optimizing the system for real-time detection in live environments with minimal delay.



**Acknowledgments**

We would like to extend our sincere gratitude to Intel Gen-AI for providing us with the resources and platform to bring this innovative project to life. Their support has been instrumental in helping us take a step closer to a world of CAPTCHA-free browsing!

Highlight of the Day!
![image](https://github.com/user-attachments/assets/0057eed4-3639-4224-83c3-f0b2a8939ebd)
