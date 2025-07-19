Real-Time Emotion Companion
This is a Python-based application built with Streamlit that serves as a real-time emotional support tool. It is designed to help users, particularly those on the autism spectrum, to better understand and recognize emotions from facial expressions and spoken language. The application uses deep learning models to provide live feedback on emotions detected from a webcam feed and microphone input.

Features
📸 Live Facial Emotion Detection: Utilizes a webcam to capture video, detect faces in real-time, and classify the facial expression into one of seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise).

🎤 Dual Voice Analysis:

🎵 Emotion from Tone: Analyzes the acoustic properties (tone, pitch) of a user's voice to predict the underlying emotion.

🗣️ Emotion from Meaning: Converts the user's speech to text and performs a basic keyword analysis to determine the emotion based on the words spoken.

🔊 Automated Spoken Feedback: When an emotion is detected from the spoken words, the application uses the browser's built-in text-to-speech engine to provide a supportive and context-aware verbal response.

Interactive UI: Built with Streamlit for a clean, simple, and interactive user experience.

How to Run
Prerequisites
Python 3.8+

A webcam connected to your computer.

A microphone connected to your computer.

1. Clone the Repository
First, clone this repository to your local machine:

git clone <your-repository-url>
cd <your-repository-directory>

2. Set Up the Environment
It is highly recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Place the Models
You must have your pre-trained models in the correct directory structure. Create a dataset folder in the root of the project and place your models as follows:

.
├── models/
│   ├── facial emotion/
│   │   └── face_emotion_model_savedmodel/
│   │       ├── assets/
│   │       ├── variables/
│   │       └── saved_model.pb
│   └── voice emotion/
│       └── Speech-Emotion-Recognition-Model/
│           ├── assets/
│           ├── variables/
│           └── saved_model.pb
├── app.py
└── requirements.txt

5. Run the Application
Once the setup is complete, run the Streamlit application from your terminal:

streamlit run app.py

Your web browser will automatically open a new tab with the running application.

Project Structure
├── dataset/
│   ├── facial emotion/
│   │   └── face_emotion_model_savedmodel/  # Directory for the face detection model
│   └── voice emotion/
│       └── Speech-Emotion-Recognition-Model/ # Directory for the voice tone model
├── app.py                      # The main Streamlit application script
└── requirements.txt            # A list of all Python dependencies

Dependencies
This project relies on the following Python libraries:

streamlit

tensorflow

numpy

opencv-python

librosa

sounddevice

SpeechRecognition

PyAudio
