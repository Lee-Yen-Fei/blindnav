import streamlit as st
import requests
from PIL import Image
import cv2
import os
from dotenv import load_dotenv
import speech_recognition as sr
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv11 import
from openai import OpenAI
import pyttsx3

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
api_key = os.getenv("AI_ML_API_KEY")
base_url = "https://api.aimlapi.com"

# Load YOLOv10 tracking model
model = YOLO('yolov10s.pt')  # Load YOLOv10 for tracking

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# Check if API keys are loaded correctly
if not api_key:
    st.error("API key is not set in the environment variables.")

# Function to convert speech to text using OpenAI Whisper API
def speech_to_text(audio_data):
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    with open(audio_data, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(base_url, headers=headers, files=files)

        if response.status_code == 200:
            return response.json()["text"]
        else:
            st.error(f"Whisper API error: {response.status_code}")
            return None

# Function to capture voice input and convert to text
def capture_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your question. Please speak now...")
        audio = recognizer.listen(source)

        try:
            st.write("Processing your question...")
            user_input = recognizer.recognize_google(audio)
            return user_input
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")

# Text-to-Speech (Offline with pyttsx3)
def text_to_speech_pyttsx3(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# Function to combine two images into a short video
def create_video_from_images(image1, image2, output_path="output_video.mp4"):
    image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

    height, width, layers = image1.shape

    image2 = resize_image(image2, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    video = cv2.VideoWriter(output_path, fourcc, 1, (width, height))  # 1 frame per second

    video.write(image1)
    video.write(image2)

    video.release()
    cv2.destroyAllWindows()

    return output_path

# Function to perform object tracking with YOLOv11
def track_objects(images):
    results = model.track(source='output_video.mp4')  # Use the tracking method
    return results  # Returns tracking results including object IDs

# Function to summarize tracked objects
def summarize_tracked_objects(results):
    static_summary = {}
    dynamic_summary = {}

    for result in results:
        for obj in result.boxes.data:  # Access tracked boxes
            obj_id = int(obj[-1])  # Get the object ID
            obj_name = result.names[obj_id]  # Get the object name
            direction = "moving" if obj[-2] else "static"  # Check if moving or static

            if direction == "moving":
                if obj_name in dynamic_summary:
                    dynamic_summary[obj_name] += 1  # Count moving objects
                else:
                    dynamic_summary[obj_name] = 1
            else:
                if obj_name in static_summary:
                    static_summary[obj_name] += 1  # Count static objects
                else:
                    static_summary[obj_name] = 1

    return static_summary, dynamic_summary

# Function to analyze objects and interact with OpenAI API
def analyze_objects(image1, image2, user_input):
    # Combine images into a video
    video_path = create_video_from_images(image1, image2)

    # Track objects in the video
    tracking_results = track_objects([video_path])

    # Summarize tracked objects
    static_summary, dynamic_summary = summarize_tracked_objects(tracking_results)

    prompt = (
        f"Static objects detected: {static_summary}. "
        f"Dynamic objects detected: {dynamic_summary}. "
        f"Within 100 words, briefly answer the blind user\'s question:{user_input}"
    )

    # Set up API call for OpenAI o1-mini chat completion
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "o1-mini",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(base_url + "/chat/completions", headers=headers, json=data)

    if response.status_code == 201:
        # Extract and return the generated message
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"Error: {response.status_code} - {response.json()}")
        return None

# Streamlit app structure
st.title("BlindNav")
st.write("Real-Time Object Detection and Navigation App")

# Upload the two images
uploaded_image_1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
uploaded_image_2 = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

if uploaded_image_1 and uploaded_image_2:
    image_1 = Image.open(uploaded_image_1)
    image_2 = Image.open(uploaded_image_2)

    st.image([image_1, image_2], caption=["Image 1", "Image 2"], use_column_width=True)

    # Capture voice input for the question
    if st.button("Ask Question"):
        user_input = capture_voice()
        if user_input:
            st.write(f"Your question: {user_input}")
            analysis = analyze_objects(image_1, image_2, user_input)
            if analysis:
                st.write(f"Analysis: {analysis}")
                text_to_speech_pyttsx3(analysis)
