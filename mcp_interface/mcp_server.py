#!/usr/bin/env python3
"""MCP Server for Semantic Video Analysis.

This server provides tools for analyzing video content and extracting semantic information
through frame selection and analysis strategies using a chain of handlers pattern.
"""

import asyncio
import os
import sys
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from handlers import HandlerChain, AnalyzeVideoHandler, AnalyzeAudioHandler
from semantic_video_analysis.models.blip_model import BlipModel
from PIL import Image


# Create the server instance
server = Server("semantic-video-analysis")

# Global BLIP model instance (initialized at startup)
blip_model = None
handler_chain = None


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


async def initialize_application():
    """Initialize the application resources."""
    global blip_model, handler_chain
    
    # Initialize BLIP model
    blip_model = BlipModel(model_name="Salesforce/blip-image-captioning-large")
    blip_model.__enter__()  # Initialize the model
    
    # Create frame analysis function and handler chain
    frame_analysis_fn = create_frame_analysis_fn()
    handler_chain = HandlerChain.of(
        AnalyzeVideoHandler(frame_analysis_fn=frame_analysis_fn)
    ).chain(
        AnalyzeAudioHandler()
    )


async def cleanup_application():
    """Cleanup application resources."""
    global blip_model
    
    if blip_model is not None:
        blip_model.__exit__(None, None, None)  # Cleanup the model
        blip_model = None


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools from all handlers in the chain."""
    return handler_chain.get_all_tools()


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    """Handle tool calls by dispatching to the appropriate handler in the chain."""
    return await handler_chain.dispatch(name, arguments)


async def main():
    """Main entry point for the MCP server."""
    try:
        # Initialize application resources
        await initialize_application()
        
        # Run the server using stdin/stdout streams
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="semantic-video-analysis",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        # Cleanup application resources
        await cleanup_application()


if __name__ == "__main__":
    asyncio.run(main())