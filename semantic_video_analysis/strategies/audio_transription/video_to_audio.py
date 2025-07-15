
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


class VideoToAudioTranslator:
    """
    A class to handle video to text transcription using the Whisper model.
    It extracts audio from a video file and transcribes it to text, automatically detecting the language.
    """
    
    def __init__(self, video_path, audio_output_path="temp_audio.wav"):
        """
        Extracts audio from a video file.

        Args:
            video_path (str): The path to the input video file (e.g., .mp4, .avi, .mkv, etc.).
            audio_output_path (str): The path to save the extracted audio file.

        Returns:
            str: The path to the audio file
        """
        video = VideoFileClip(video_path)
        # Extract audio and save it to the specified output path
        video.audio.write_audiofile(audio_output_path, verbose=False, logger=None)
