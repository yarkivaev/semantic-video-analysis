#!/usr/bin/env python3
"""MCP Server for Semantic Video Analysis.

This server provides tools for analyzing video content and extracting semantic information
through frame selection and analysis strategies using a chain of handlers pattern.
"""

import asyncio
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from handlers import HandlerChain, AnalyzeVideoHandler


# Create the server instance
server = Server("semantic-video-analysis")

# Initialize handler chain (immutable)
handler_chain = HandlerChain.of(AnalyzeVideoHandler())


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


if __name__ == "__main__":
    asyncio.run(main())