import streamlit as st
import requests
from PIL import Image
import cv2
import os
from dotenv import load_dotenv
import speech_recognition as sr
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv10 import
from openai import OpenAI
from gtts import gTTS  # Import gTTS for text-to-speech

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
def speech_to_text_via_url(audio_url):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    url = f"{base_url}/stt"  # Ensure this endpoint is correct

    data = {
        "url": audio_url,
        "model": "#g1_whisper-large",  # Ensure the correct model name
    }

    # Make a POST request to the STT API with the audio URL
    response = requests.post(url, headers=headers, json=data)

    # Handle response
    if response.status_code >= 400:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None
    else:
        response_data = response.json()
        transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
        st.write("[transcription]", transcript)
        return transcript

# Text-to-Speech function using gTTS
def text_to_speech_gtts(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)  # Save the audio file
    return audio_file  # Return the file path

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
        f"Within 100 words, briefly answer the blind user's question: {user_input}"
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

# Input field for the URL to the audio file
audio_url = st.text_input("Enter the URL of the audio file")

if uploaded_image_1 and uploaded_image_2:
    image_1 = Image.open(uploaded_image_1)
    image_2 = Image.open(uploaded_image_2)

    st.image([image_1, image_2], caption=["Image 1", "Image 2"], use_column_width=True)

if audio_url:
    user_input = speech_to_text_via_url(audio_url)

    if user_input:
        st.write(f"Your question: {user_input}")

        # Analyze images based on the user's question
        analysis = analyze_objects(image_1, image_2, user_input)

        if analysis:
            st.write(f"Analysis: {analysis}")
            audio_path = text_to_speech_gtts(analysis)  # Convert analysis to speech
            st.audio(audio_path, format='audio/mp3')  # Play the generated audio
else:
    st.error("Please enter an audio URL.")
