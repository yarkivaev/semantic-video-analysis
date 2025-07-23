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


class AnalyzeVideoHandler(BaseHandler):
    """Handler for the analyze_video tool."""
    
    def __init__(self, frame_analysis_fn=None, enable_technical_analysis=True):
        """Initialize the handler with a frame analysis function.
        
        Args:
            frame_analysis_fn: Function that takes a frame path and returns analysis text.
                              If None, a default placeholder function will be used.
            enable_technical_analysis: Whether to enable technical analysis of frames.
        """
        self.frame_analysis_fn = frame_analysis_fn
        self.enable_technical_analysis = enable_technical_analysis
    
    def get_tool_definition(self) -> types.Tool:
        """Return the tool definition for video analysis."""
        return types.Tool(
            name="analyze_video_with_technical",
            description="Analyze video content with advanced technical analysis and semantic information from frames",
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
                    },
                    "enable_technical_analysis": {
                        "type": "boolean",
                        "description": (
                            "Enable comprehensive technical analysis of video frames including: \n"
                            "- image quality metrics (clarity, contrast, brightness, sharpness, saturation), \n"
                            "- color analysis (balance, distribution, temperature), \n"
                            "- lens type classification. \n"
                            "- There is also a qualitative characterization, which is based on scalar metrics and is rather inaccurate. \n"
                            "Quality labels are assigned using fixed thresholds: "
                            "- clarity/sharpness >400=good, >200=average, else=bad; \n"
                            "- contrast >40=good, >20=average, else=bad; \n"
                            "- brightness 0.3-0.7=good, 0.2-0.3 or 0.7-0.8=average, else=bad; \n"
                            "- saturation 0.4-0.7=good, 0.2-0.4=average, else=bad; \n"
                            "- color temperature -0.2 to 0.2=neutral, 0.2-0.3=probably_warm, >0.3=warm, -0.3 to -0.2=probably_cold, <-0.3=cold. \n"
                            "(default: true)"
                        ),
                        "default": True
                    }
                },
                "required": ["video_path"]
            }
        )
    
    def can_handle(self, tool_name: str) -> bool:
        """Check if this handler can handle the analyze_video_with_technical tool."""
        return tool_name == "analyze_video_with_technical"
    
    async def handle(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Handle the analyze_video tool call."""
        if not arguments:
            raise ValueError("Missing arguments for analyze_video")
        
        video_path = arguments.get("video_path")
        period = arguments.get("period", 1.0)
        enable_technical_analysis = arguments.get("enable_technical_analysis", self.enable_technical_analysis)
        
        if not video_path:
            raise ValueError("video_path is required")
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        try:
            # Use injected frame analysis function or default placeholder
            if self.frame_analysis_fn is None:
                def frame_analysis_fn(frame_path: str) -> str:
                    return f"Frame analysis for {os.path.basename(frame_path)}"
            else:
                frame_analysis_fn = self.frame_analysis_fn
            
            # Create periodic selection strategy
            strategy = PeriodicSelectionStrategy.from_video_file(video_path, period)
            
            # Create and run analysis
            analyzer = FrameSelectionAnalysis(
                video_path, 
                frame_analysis_fn, 
                strategy,
                enable_technical_analysis=enable_technical_analysis
            )
            media_context = analyzer.analyse()
            
            # Format the results
            result = {
                "video_path": video_path,
                "analysis_period": period,
                "total_actions": len(media_context.actions),
                "actions": []
            }
            
            for action in media_context.actions:
                action_result = {
                    "start": action.start,
                    "end": action.end,
                    "duration": action.end - action.start,
                    "content": action.content
                }
                
                # Add technical analysis if available
                if hasattr(action, 'technical_analysis') and action.technical_analysis:
                    action_result["technical_analysis"] = action.technical_analysis
                
                result["actions"].append(action_result)
            
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