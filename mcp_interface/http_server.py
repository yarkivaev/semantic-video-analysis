#!/usr/bin/env python3
"""HTTP MCP Server for Semantic Video Analysis.

This server provides MCP tools via HTTP transport for analyzing video content
and extracting semantic information.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mcp_interface.handlers import HandlerChain, AnalyzeVideoHandler, AnalyzeAudioHandler, ListVideosHandler
from semantic_video_analysis.models.blip_model import BlipModel
from semantic_video_analysis.strategies.audio_transription.audio_to_text import AudioToTextTranslator
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global resources
blip_model = None
whisper_model = None
handler_chain = None


class MCPRequest(BaseModel):
    """MCP request model."""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None


class MCPToolCall(BaseModel):
    """MCP tool call parameters."""
    name: str
    arguments: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP response model."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None


def create_frame_analysis_fn():
    """Create a frame analysis function that uses the global BLIP model."""
    def frame_analysis_fn(frame_path: str) -> str:
        try:
            image = Image.open(frame_path)
            caption = blip_model.generate_caption(image)
            return caption
        except Exception as e:
            return f"Frame analysis failed for {os.path.basename(frame_path)} (Error: {str(e)})"
    
    return frame_analysis_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global blip_model, whisper_model, handler_chain
    
    # Startup
    logger.info("Initializing BLIP model...")
    blip_model = BlipModel(model_name="Salesforce/blip-image-captioning-large")
    blip_model.__enter__()  # Initialize the model
    
    logger.info("Initializing Whisper model...")
    # Detect device for Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for Whisper: {device}")
    
    # Initialize Whisper model (base model for faster loading)
    whisper_model = AudioToTextTranslator(model_name="base", device=device)
    logger.info("Whisper model loaded successfully")
    
    # Create frame analysis function and handler chain
    frame_analysis_fn = create_frame_analysis_fn()
    handler_chain = HandlerChain.of(
        AnalyzeVideoHandler(frame_analysis_fn=frame_analysis_fn),
        AnalyzeAudioHandler(audio_translator=whisper_model),
        ListVideosHandler()
    )
    logger.info("MCP server initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP server...")
    if blip_model is not None:
        blip_model.__exit__(None, None, None)
        blip_model = None
    whisper_model = None


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Semantic Video Analysis MCP Server",
    description="MCP server for video and audio analysis",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "semantic-video-analysis",
        "model_loaded": blip_model is not None
    }


@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle MCP protocol requests."""
    try:
        # Handle different MCP methods
        if request.method == "initialize":
            # Initialize session
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "semantic-video-analysis",
                    "version": "1.0.0"
                }
            }
            return MCPResponse(id=request.id, result=result)
        
        elif request.method == "tools/list":
            # List available tools
            tools = []
            for tool in handler_chain.get_all_tools():
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                })
            result = {"tools": tools}
            return MCPResponse(id=request.id, result=result)
        
        elif request.method == "tools/call":
            # Execute tool
            if not request.params:
                raise ValueError("Missing params for tools/call")
            
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            if not tool_name:
                raise ValueError("Tool name is required")
            
            # Dispatch to handler chain
            tool_results = await handler_chain.dispatch(tool_name, arguments)
            
            # Format results for MCP response
            content = []
            for result in tool_results:
                if result.type == "text":
                    content.append({
                        "type": "text",
                        "text": result.text
                    })
            
            result = {"content": content}
            return MCPResponse(id=request.id, result=result)
        
        elif request.method == "ping":
            # Simple ping response
            return MCPResponse(id=request.id, result={"pong": True})
        
        else:
            # Unknown method
            error = {
                "code": -32601,
                "message": f"Method not found: {request.method}"
            }
            return MCPResponse(id=request.id, error=error)
    
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        error = {
            "code": -32603,
            "message": str(e)
        }
        return MCPResponse(id=request.id, error=error)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Semantic Video Analysis MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "/mcp": "MCP protocol endpoint (POST)",
            "/health": "Health check endpoint (GET)",
            "/": "Service information (GET)"
        },
        "tools": [
            "analyze_video - Analyze video frames and extract semantic information",
            "analyze_audio - Transcribe audio from video files"
        ]
    }


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8090"))
    
    # Run the server
    uvicorn.run(
        "http_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )