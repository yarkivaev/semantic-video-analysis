#!/usr/bin/env python3

from semantic_video_analysis.strategies.audio_transription.video_to_audio import VideoToAudioTranslator
from semantic_video_analysis.strategies.audio_transription.audio_to_text import AudioToTextTranslator
import os
import torch

def main():
    """
    Test script that transcribes speech from tg_video2.mp4 using VideoToAudioTranslator and AudioToTextTranslator classes.
    """
    
    # Define file paths
    video_file = "examples/Chocolate2_low.mp4" # change for your video
    audio_file = "extracted_audio.wav"
    
    # Check if video file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found in current directory.")
        return
    
    print(f"Processing video: {video_file}")
    
    # Step 1: Extract audio from video
    print("Step 1: Extracting audio from video...")
    video_to_audio = VideoToAudioTranslator(video_file, audio_file)
    print(f"Audio extracted and saved to: {audio_file}")
    
    # Step 2: Transcribe audio to text
    print("Step 2: Transcribing audio to text...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    audio_to_text = AudioToTextTranslator(model_name="large-v3", device=device)
    transcribed_text = audio_to_text.transcribe_audio(audio_file)
    
    print("\n" + "="*50)
    print("TRANSCRIPTION RESULTS")
    print("="*50)
    print(f"Transcribed Text:\n{transcribed_text}")
    print("="*50)
    
    # Clean up temporary audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"\nCleaned up temporary audio file: {audio_file}")

if __name__ == "__main__":
    main()