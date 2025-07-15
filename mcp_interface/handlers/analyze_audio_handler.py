"""Handler for audio analysis tool."""

import json
import os
import tempfile
import torch
from typing import Any

import mcp.types as types
from .base_handler import BaseHandler

# Import from the parent directory to access the semantic_video_analysis package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from semantic_video_analysis.strategies.audio_transription.video_to_audio import VideoToAudioTranslator
from semantic_video_analysis.strategies.audio_transription.audio_to_text import AudioToTextTranslator


class AnalyzeAudioHandler(BaseHandler):
    """Handler for the analyze_audio tool."""
    
    def __init__(self):
        """Initialize the handler."""
        super().__init__()
    
    def get_tool_definition(self) -> types.Tool:
        """Return the tool definition for audio analysis."""
        return types.Tool(
            name="analyze_audio",
            description="Extract and transcribe audio from video files using Whisper model",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file to analyze"
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Whisper model name to use (default: large-v3)",
                        "default": "large-v3"
                    }
                },
                "required": ["video_path"]
            }
        )
    
    def can_handle(self, tool_name: str) -> bool:
        """Check if this handler can handle the analyze_audio tool."""
        return tool_name == "analyze_audio"
    
    async def handle(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Handle the analyze_audio tool call."""
        if not arguments:
            raise ValueError("Missing arguments for analyze_audio")
        
        video_path = arguments.get("video_path")
        model_name = arguments.get("model_name", "large-v3")
        
        if not video_path:
            raise ValueError("video_path is required")
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Create temporary file for extracted audio
        temp_audio_file = None
        
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_audio_file = temp_file.name
            
            # Step 1: Extract audio from video
            video_to_audio = VideoToAudioTranslator(video_path, temp_audio_file)
            
            # Step 2: Transcribe audio to text
            # Auto-detect device (GPU if available, otherwise CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            audio_to_text = AudioToTextTranslator(model_name=model_name, device=device)
            transcription_segments = audio_to_text.transcribe_audio(temp_audio_file)
            
            # Format the results
            result = {
                "video_path": video_path,
                "model_name": model_name,
                "device_used": device,
                "segments": transcription_segments,
                "total_segments": len(transcription_segments),
                "audio_extracted_successfully": True
            }
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )
            ]
            
        except Exception as e:
            return [
                types.TextContent(
                    type="text", 
                    text=f"Error analyzing audio: {str(e)}"
                )
            ]
        
        finally:
            # Clean up temporary audio file
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                except OSError:
                    pass  # Ignore cleanup errors