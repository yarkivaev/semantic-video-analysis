"""Integration tests for AnalyzeVideoHandler."""

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_interface.handlers.analyze_video_handler import AnalyzeVideoHandler
from semantic_video_analysis.models.blip_model import BlipModel
import mcp.types as types


class TestAnalyzeVideoHandlerIntegration:
    """Integration tests for the AnalyzeVideoHandler using real video files."""
    
    @pytest.fixture
    def blip_frame_analysis_fn(self):
        """Create a BLIP-based frame analysis function using base model."""
        with BlipModel(model_name="Salesforce/blip-image-captioning-base") as blip_model:
            def frame_analysis_fn(frame_path: str) -> str:
                try:
                    image = Image.open(frame_path)
                    caption = blip_model.generate_caption(image)
                    return caption
                except Exception as e:
                    return f"Frame analysis failed for {os.path.basename(frame_path)} (Error: {str(e)})"
            
            yield frame_analysis_fn
    
    @pytest.fixture
    def handler(self, blip_frame_analysis_fn):
        """Create an AnalyzeVideoHandler instance with BLIP analysis."""
        return AnalyzeVideoHandler(frame_analysis_fn=blip_frame_analysis_fn)
    
    @pytest.fixture
    def handler_no_blip(self):
        """Create an AnalyzeVideoHandler instance without BLIP for basic tests."""
        return AnalyzeVideoHandler()
    
    @pytest.fixture
    def sample_video_path(self):
        """Get path to a sample video file from examples."""
        examples_dir = project_root / "examples"
        video_path = examples_dir / "Business_center_low.mp4"
        
        if not video_path.exists():
            pytest.skip(f"Sample video not found at {video_path}")
        
        return str(video_path)
    
    def test_tool_definition(self, handler_no_blip):
        """Test that the tool definition is correct."""
        tool_def = handler_no_blip.get_tool_definition()
        
        assert isinstance(tool_def, types.Tool)
        assert tool_def.name == "analyze_video"
        assert "video content" in tool_def.description.lower()
        
        # Check input schema
        schema = tool_def.inputSchema
        assert schema["type"] == "object"
        assert "video_path" in schema["properties"]
        assert "period" in schema["properties"]
        assert "video_path" in schema["required"]
        
        # Ensure frame_analysis_description is not present
        assert "frame_analysis_description" not in schema["properties"]
    
    def test_can_handle(self, handler_no_blip):
        """Test the can_handle method."""
        assert handler_no_blip.can_handle("analyze_video") is True
        assert handler_no_blip.can_handle("other_tool") is False
    
    @pytest.mark.asyncio
    async def test_analyze_video_integration(self, handler, sample_video_path):
        """Test full video analysis with BLIP model integration."""
        # Test arguments
        arguments = {
            "video_path": sample_video_path,
            "period": 2.0  # Analyze every 2 seconds
        }
        
        # Run the analysis
        result = await handler.handle("analyze_video", arguments)
        
        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert result[0].type == "text"
        
        # Parse the JSON result
        result_data = json.loads(result[0].text)
        
        # Verify result content
        assert "video_path" in result_data
        assert "analysis_period" in result_data
        assert "total_actions" in result_data
        assert "actions" in result_data
        
        assert result_data["video_path"] == sample_video_path
        assert result_data["analysis_period"] == 2.0
        assert isinstance(result_data["total_actions"], int)
        assert result_data["total_actions"] > 0
        assert isinstance(result_data["actions"], list)
        
        # Verify action structure
        for action in result_data["actions"]:
            assert "start" in action
            assert "end" in action
            assert "duration" in action
            assert "content" in action
            
            # Verify types
            assert isinstance(action["start"], (int, float))
            assert isinstance(action["end"], (int, float))
            assert isinstance(action["duration"], (int, float))
            assert isinstance(action["content"], str)
            
            # Verify logical constraints
            assert action["end"] > action["start"]
            assert action["duration"] == action["end"] - action["start"]
            
            # Verify content is meaningful (not just placeholder)
            assert len(action["content"]) > 0
            assert "Error:" not in action["content"]  # Should not have errors in normal case
    
    @pytest.mark.asyncio
    async def test_analyze_video_with_default_period(self, handler, sample_video_path):
        """Test video analysis with default period."""
        arguments = {
            "video_path": sample_video_path
        }
        
        result = await handler.handle("analyze_video", arguments)
        result_data = json.loads(result[0].text)
        
        # Should use default period of 1.0
        assert result_data["analysis_period"] == 1.0
        assert result_data["total_actions"] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_video_nonexistent_file(self, handler_no_blip):
        """Test handling of nonexistent video file."""
        arguments = {
            "video_path": "/nonexistent/path/video.mp4"
        }
        
        result = await handler_no_blip.handle("analyze_video", arguments)
        
        # Should return error message
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "not found" in result[0].text.lower() or "error" in result[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_video_missing_arguments(self, handler_no_blip):
        """Test handling of missing arguments."""
        with pytest.raises(ValueError, match="Missing arguments"):
            await handler_no_blip.handle("analyze_video", None)
    
    @pytest.mark.asyncio
    async def test_analyze_video_missing_video_path(self, handler_no_blip):
        """Test handling of missing video_path argument."""
        arguments = {
            "period": 1.0
        }
        
        with pytest.raises(ValueError, match="video_path is required"):
            await handler_no_blip.handle("analyze_video", arguments)
    
    @pytest.mark.asyncio
    async def test_multiple_videos(self, handler):
        """Test analysis of multiple different videos."""
        examples_dir = project_root / "examples"
        video_files = list(examples_dir.glob("*.mp4"))
        
        if len(video_files) < 2:
            pytest.skip("Need at least 2 video files for this test")
        
        results = []
        for video_path in video_files[:2]:  # Test first 2 videos
            arguments = {
                "video_path": str(video_path),
                "period": 3.0
            }
            
            result = await handler.handle("analyze_video", arguments)
            result_data = json.loads(result[0].text)
            results.append(result_data)
        
        # Verify each video was analyzed
        for i, result_data in enumerate(results):
            assert result_data["video_path"] == str(video_files[i])
            assert result_data["total_actions"] > 0
            
        # Results should be different (different content for different videos)
        if len(results) == 2:
            assert results[0]["actions"] != results[1]["actions"]