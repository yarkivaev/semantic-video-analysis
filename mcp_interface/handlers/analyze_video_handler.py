import json
import os
from typing import Any
import cv2
import numpy as np

import mcp.types as types
from .base_handler import BaseHandler

# Import from the parent directory to access the semantic_video_analysis package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from semantic_video_analysis.models.tech_analysis import analyze_video, detect_scenes


class AnalyzeVideoHandler(BaseHandler):
    """Handler for the analyze_video tool."""
    
    def __init__(self, frame_analysis_fn=None):
        """Initialize the handler with a frame analysis function.
        
        Args:
            frame_analysis_fn: Function that takes a frame path and returns analysis text.
                              If None, a default placeholder function will be used.
        """
        self.frame_analysis_fn = frame_analysis_fn
    
    def get_tool_definition(self) -> types.Tool:
        """Return the tool definition for video analysis."""
        return types.Tool(
            name="analyze_video",
            description="Analyze video content, extract semantic information from frames, and compute scene metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file to analyze"
                    },
                    "svm_model_path": {
                        "type": "string",
                        "description": "Path to the SVM model file for shot type prediction (default: svm_model.joblib)",
                        "default": "svm_model.joblib"
                    }
                },
                "required": ["video_path"]
            }
        )
    
    def can_handle(self, tool_name: str) -> bool:
        """Check if this handler can handle the analyze_video tool."""
        return tool_name == "analyze_video"
    
    async def handle(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Handle the analyze_video tool call."""
        if not arguments:
            raise ValueError("Missing arguments for analyze_video")
        
        video_path = arguments.get("video_path")
        svm_model_path = arguments.get("svm_model_path", "svm_model.joblib")
        
        if not video_path:
            raise ValueError("video_path is required")
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if not os.path.exists(svm_model_path):
            raise ValueError(f"SVM model file not found: {svm_model_path}")
        
        try:
            # Run video analysis to get scene metrics
            analysis_result = analyze_video(video_path, svm_model_path)
            
            # Use injected frame analysis function or default placeholder
            if self.frame_analysis_fn is None:
                def frame_analysis_fn(frame_path: str) -> str:
                    return f"Frame analysis for {os.path.basename(frame_path)}"
            
            # Process each scene to extract frames and generate captions
            scenes = analysis_result["scenes"]
            for scene in scenes:
                start_time = scene["start_time"]
                end_time = scene["end_time"]
                
                # Extract frames at 1-second intervals within the scene
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_interval = fps  # 1 second
                frame_captions = []
                
                current_time = start_time
                while current_time < end_time:
                    cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Save frame to temporary file
                    temp_frame_path = f"temp_frame_{int(current_time*1000)}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Analyze frame
                    caption = self.frame_analysis_fn(temp_frame_path)
                    frame_captions.append({
                        "timestamp": current_time,
                        "caption": caption
                    })
                    
                    # Clean up temporary frame
                    os.remove(temp_frame_path)
                    
                    current_time += 1.0
                
                cap.release()
                scene["frame_captions"] = frame_captions
            
            # Format the results
            result = {
                "video_path": video_path,
                "scenes": scenes
            }
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )
            ]
            
        except Exception as e:
            return [
                types.TextContent(
                    type="text", 
                    text=f"Error analyzing video: {str(e)}"
                )
            ]