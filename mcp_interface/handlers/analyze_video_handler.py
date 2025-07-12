"""Handler for video analysis tool."""

import json
import os
from typing import Any

import mcp.types as types
from .base_handler import BaseHandler

# Import from the parent directory to access the semantic_video_analysis package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from semantic_video_analysis.strategies import FrameSelectionAnalysis, PeriodicSelectionStrategy
from semantic_video_analysis.models.blip_model import BlipModel
from PIL import Image


class AnalyzeVideoHandler(BaseHandler):
    """Handler for the analyze_video tool."""
    
    def get_tool_definition(self) -> types.Tool:
        """Return the tool definition for video analysis."""
        return types.Tool(
            name="analyze_video",
            description="Analyze video content and extract semantic information from frames",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file to analyze"
                    },
                    "period": {
                        "type": "number",
                        "description": "Time period in seconds between frame selections (default: 1.0)",
                        "default": 1.0
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
        period = arguments.get("period", 1.0)
        
        if not video_path:
            raise ValueError("video_path is required")
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        try:
            # Initialize BLIP model
            with BlipModel() as blip_model:
                # Create frame analysis function using BLIP
                def frame_analysis_fn(frame_path: str) -> str:
                    try:
                        # Load the frame image
                        image = Image.open(frame_path)
                        # Generate caption using BLIP
                        caption = blip_model.generate_caption(image)
                        return caption
                    except Exception as e:
                        # Fallback to generic description if BLIP fails
                        return f"Frame analysis failed for {os.path.basename(frame_path)} (Error: {str(e)})"
                
                # Create periodic selection strategy
                strategy = PeriodicSelectionStrategy.from_video_file(video_path, period)
                
                # Create and run analysis
                analyzer = FrameSelectionAnalysis(video_path, frame_analysis_fn, strategy)
                media_context = analyzer.analyse()
                
                # Format the results
                result = {
                    "video_path": video_path,
                    "analysis_period": period,
                    "total_actions": len(media_context.actions),
                    "actions": []
                }
                
                for action in media_context.actions:
                    result["actions"].append({
                        "start": action.start,
                        "end": action.end,
                        "duration": action.end - action.start,
                        "content": action.content
                    })
                
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
                    text=f"Error analyzing video: {str(e)}"
                )
            ]