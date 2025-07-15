# !pip install -q git+https://github.com/openai/whisper.git
# !sudo apt-get install -y ffmpeg


import whisper
import os
import cv2
import torch
import json
import wave
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("extracted_frames", exist_ok=True)

whisper_model = whisper.load_model("large-v3")


def transcribe_audio_whisper(video_path, audio_output_path):
    """
    Transcribes the audio from a video file to text using the Whisper model
    and saves the audio to a specified file. Language is automatically detected.

    Args:
        video_path (str): The path to the input video file (e.g., .mp4, .avi, .mkv, etc.).
        audio_output_path (str): The path to save the extracted audio file.

    Returns:
        tuple: A tuple containing:
            - str: The transcribed text.
            - list: A list of segment dictionaries (from Whisper's output),
                    or an empty list if not available.
            - str: The detected language code.
    """
    video = VideoFileClip(video_path)
    # Extract audio and save it to the specified output path
    video.audio.write_audiofile(audio_output_path, verbose=False, logger=None)

    # Transcribe the audio with automatic language detection
    result = whisper_model.transcribe(audio_output_path)

    return result["text"], result.get("segments", []), result["language"]



# Example usage
video_path = "tg_video_13_07.mp4"  # Example: "my_video.avi", "another_video.mkv"
audio_output_path = "extracted_audio.wav"  # Specify the output audio file path

transcribed_text, segments, detected_language = transcribe_audio_whisper(video_path, audio_output_path)

print(f"\n--- Detected Language: {detected_language} ---")
print("\n--- Transcription ---")
print(transcribed_text)
print("\n--- Segments (if available) ---")
for segment in segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
