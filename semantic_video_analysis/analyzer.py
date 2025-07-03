import os
import cv2
import json
import torch
from PIL import Image
from datetime import datetime
from moviepy import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper

class VideoAnalyzer:
    """Main class for analyzing videos and generating semantic descriptions."""
    
    def __init__(self, device=None, model_name="Salesforce/blip-image-captioning-large", enable_audio=True):
        """Initialize the VideoAnalyzer with specified device and model.
        
        Args:
            device (str, optional): Device to run the model on. Defaults to auto-detect.
            model_name (str, optional): Name of the BLIP model to use.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        self.enable_audio = enable_audio
        if self.enable_audio:
          self.audio_model = whisper.load_model("base", )

        os.makedirs("extracted_frames", exist_ok=True)
    
    def extract_key_frames(self, video_path, num_frames=5):
        """Extract key frames from a video at regular intervals.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of frames to extract.
            
        Returns:
            list: Paths to the extracted frame images.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frame_paths = []
        
        for idx, frame_num in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                base = os.path.basename(video_path).split('.')[0]
                path = f"extracted_frames/{base}_frame{idx}.jpg"
                cv2.imwrite(path, frame)
                frame_paths.append(path)
        
        cap.release()
        return frame_paths
    
    def generate_caption(self, image_path):
        """Generate a caption for an image using BLIP model.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Generated caption for the image.
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs)
        
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def get_file_info(self, video_path):
        """Extract file system information about the video.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            tuple: File size in bytes and creation timestamp.
        """
        stat = os.stat(video_path)
        size = stat.st_size
        created = datetime.utcfromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S UTC')
        return size, created
    
    def transcribe_audio(self, video_path):
        """Extract and transcribe audio from a video using Whisper."""
        try:
            # Whisper can take the video path directly
            result = self.audio_model.transcribe(video_path)
            return result["text"]
        except Exception as e:
            print(f"Audio transcription failed: {e}")
            return ""

    def describe_video(self, video_path, num_frames=5):
        """Generate a comprehensive semantic description of a video.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of frames to analyze.
            
        Returns:
            dict: Structured JSON object with video description.
        """
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            width, height = clip.size
            fps = clip.fps
            has_audio = clip.audio is not None
            video_format = video_path.split('.')[-1]
            clip.close()
        except Exception as e:
            raise RuntimeError(f"Video metadata extraction failed: {e}")
        
        # Metadata
        file_name = os.path.basename(video_path)
        file_path = os.path.abspath(video_path)
        file_size, created_at = self.get_file_info(video_path)
        
        # Captions
        frame_paths = self.extract_key_frames(video_path, num_frames=num_frames)
        frame_captions = [self.generate_caption(fp) for fp in frame_paths]
        
        audio_transcript = ""
        if self.enable_audio and has_audio:
          print("Transcribing audio...")
          audio_transcript = self.transcribe_audio(video_path)

        # Extract tags from captions
        tags = list(set(word for cap in frame_captions for word in cap.lower().split() 
                       if word.isalpha() and len(word) > 3))[:5]
        
        # Build structured JSON
        json_obj = {
            "type": "video",
            "metadata": {
                "fileName": file_name,
                "filePath": file_path,
                "fileSize": file_size,
                "createdAt": created_at,
                "description": " ".join(set(frame_captions)),
                # "description": " ".join(set(frame_captions[:2])),
                "tags": tags
            },
            "duration": duration,
            "resolution": {"width": width, "height": height},
            "frameRate": fps,
            "hasAudio": has_audio,
            "videoFormat": video_format,
            "contentAnalysis": {
                "contentOverview": " ".join(set(frame_captions)),
                "actionIntroduction": frame_captions[0],
                "timeBoundDetails": [
                    {
                        "detailStartTime": round(i * (duration / num_frames), 2),
                        "detailEndTime": round((i + 1) * (duration / num_frames), 2),
                        "detailDescription": frame_captions[i],
                        "detailConfidence": round(0.8 + 0.02 * (num_frames - i) / num_frames, 2)
                    }
                    for i in range(len(frame_captions))
                ],
                "detectedObjects": tags[:5],
                "detectedScenes": list(set([
                    "indoor" if any(word in cap.lower() for word in ["room", "bed", "table", "chair"]) 
                    else "outdoor" for cap in frame_captions
                ])),
                "estimatedMood": "neutral",
                "audioTranscript": audio_transcript if audio_transcript else None
            }
        }
        
        return json_obj
    
    def analyze_videos(self, video_paths, output_dir="."):
        """Analyze multiple videos and save their descriptions.
        
        Args:
            video_paths (list): List of paths to video files.
            output_dir (str): Directory to save output JSON files.
            
        Returns:
            list: List of description dictionaries.
        """
        all_descriptions = []
        
        for path in video_paths:
            print(f"Processing: {path}")
            desc = self.describe_video(path, num_frames=10)
            all_descriptions.append(desc)
            
            output_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(path))[0]}_description.json"
            )
            
            with open(output_file, "w") as f:
                json.dump(desc, f, indent=2)
        
        return all_descriptions
