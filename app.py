import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import librosa
import sounddevice as sd
import speech_recognition as sr
import time
import streamlit.components.v1 as components

# --- FUNCTION TO USE BROWSER'S BUILT-IN TTS ---
def speak_web(text):
    """
    Uses Streamlit components to execute JavaScript for web-based text-to-speech,
    which is more reliable than pyttsx3 in a web app environment.
    """
    # The JavaScript code creates a SpeechSynthesisUtterance object and speaks the text.
    # The autoplay and hidden attributes are used to make the audio play automatically
    # without showing any visible HTML element.
    js_code = f"""
        <script>
            var message = new SpeechSynthesisUtterance();
            message.text = "{text}";
            window.speechSynthesis.speak(message);
        </script>
    """
    # Execute the JavaScript in the Streamlit app
    components.html(js_code, height=0, width=0)

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    """Loads the face and audio emotion models."""
    face_model = None
    audio_model = None
    st.info("Loading models... Please wait.")
    try:
        face_model = tf.keras.models.load_model("models/facial emotion/face_emotion_model_savedmodel")
    except Exception as e:
        st.error(f"CRITICAL: Failed to load face model. Please check the path. Error: {e}")
    
    try:
        audio_model = tf.keras.models.load_model("models/voice emotion/Speech-Emotion-Recognition-Model")
    except Exception as e:
        st.error(f"CRITICAL: Failed to load audio model. Please check the path. Error: {e}")
        
    return face_model, audio_model

face_model, audio_model = load_models()

# --- APP CONFIGURATION ---
st.title("🧠 Real-Time Emotion Companion")
st.caption("This tool analyzes facial expressions and spoken words to provide emotional feedback.")

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad','Neutral','Surprise']

# --- FACIAL EMOTION DETECTION ---
st.header("📸 Live Facial Emotion Detection")
run_face_detection = st.checkbox("Start Webcam for Facial Emotion")

FRAME_WINDOW = st.image([])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if run_face_detection and face_model is not None:
    try:
        face_infer = face_model.signatures['serving_default']
    except Exception as e:
        st.error(f"Could not get face model signature: {e}")
        face_infer = None

    if face_infer:
        cap = cv2.VideoCapture(0)
        while run_face_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam. Please try again.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = np.reshape(roi, (1, 48, 48, 1))
                input_tensor = tf.constant(roi, dtype=tf.float32)

                try:
                    prediction_dict = face_infer(input_tensor)
                    prediction = prediction_dict[list(prediction_dict.keys())[0]]
                    emotion_index = np.argmax(prediction)
                    emotion = EMOTION_LABELS[emotion_index]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    st.error(f"Face prediction failed: {e}")
            
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

elif run_face_detection and face_model is None:
    st.error("Face model is not loaded. Cannot start webcam.")

# --- VOICE EMOTION DETECTION ---
st.header("🎤 Voice and Speech Emotion Analysis")
st.markdown("Click the button below and speak. The app will analyze both the **tone** of your voice and the **meaning** of your words.")

duration = st.slider("Select recording duration (seconds)", 2, 10, 5)
record_button = st.button("🎙️ Record and Analyze Voice")

if record_button and audio_model is not None:
    try:
        audio_infer = audio_model.signatures['serving_default']
    except Exception as e:
        st.error(f"Could not get audio model signature: {e}")
        audio_infer = None

    if audio_infer:
        samplerate = 22050
        with st.spinner(f"Listening for {duration} seconds..."):
            audio_recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
        
        st.success("Recording complete!")

        st.markdown("---")
        st.subheader("Analysis from Spoken Words")
        try:
            audio_int16 = (audio_recording * 32767).astype(np.int16)
            audio_data = sr.AudioData(audio_int16.tobytes(), samplerate, sample_width=2)
            recognizer = sr.Recognizer()
            text = recognizer.recognize_google(audio_data)
            st.write(f"**You said:** *\"{text}\"*")

            st.info("Note: Word analysis uses a simple keyword search, not a deep learning model.")
            text_lower = text.lower()
            text_emotion = "Neutral"
            if any(word in text_lower for word in ["excited", "happy", "joy", "love that", "wonderful", "great"]):
                text_emotion = "Happy"
            elif any(word in text_lower for word in ["sad", "don't feel good", "not in the mood", "upset"]):
                text_emotion = "Sad"
            elif any(word in text_lower for word in ["angry", "why did you", "don't", "furious", "hate"]):
                text_emotion = "Angry"
            elif any(word in text_lower for word in ["wow", "oh my god", "no way", "amazing", "incredible"]):
                text_emotion = "Surprise"
            elif any(word in text_lower for word in ["scared", "afraid", "fear"]):
                text_emotion = "Fear"
            
            st.markdown(f"### 🗣️ Emotion from Meaning: **{text_emotion}**")

            emotion_responses = {
                'Sad': "I hear that you might be feeling sad. Please know that your feelings are valid. Take a moment for yourself.",
                'Happy': "It sounds like you're happy! That's wonderful to hear.",
                'Angry': "It seems you might be feeling angry. It's okay to feel that way. Taking a deep breath can sometimes help.",
                'Surprise': "Oh, it sounds like something surprised you!",
                'Fear': "It sounds like you might be scared. Remember, you are in a safe space."
            }
            
            if text_emotion in emotion_responses:
                response_message = emotion_responses[text_emotion]
                st.info(f"Speaking response: \"{response_message}\"")
                # Use the new web-based TTS function
                speak_web(response_message)

        except sr.UnknownValueError:
            st.warning("Could not understand the words spoken. Please try speaking more clearly.")
        except sr.RequestError as e:
            st.error(f"Could not request results from the speech recognition service; {e}")
        except Exception as e:
            st.error(f"Word analysis failed: {e}")

elif record_button and audio_model is None:
    st.error("Audio model is not loaded. Cannot perform voice analysis.")
