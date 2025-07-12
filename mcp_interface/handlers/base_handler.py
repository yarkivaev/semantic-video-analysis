"""Base handler interface for MCP tools."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import mcp.types as types


class BaseHandler(ABC):
    """Abstract base class for MCP tool handlers."""
    
    def __init__(self, next_handler: Optional['BaseHandler'] = None):
        """Initialize handler with optional next handler in chain.
        
        Args:
            next_handler: Next handler in the chain of responsibility.
        """
        self._next_handler = next_handler
    
    def set_next(self, handler: 'BaseHandler') -> 'BaseHandler':
        """Set the next handler in the chain.
        
        Args:
            handler: The next handler to set.
            
        Returns:
            The handler that was set, for method chaining.
        """
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def get_tool_definition(self) -> types.Tool:
        """Return the tool definition for this handler.
        
        Returns:
            Tool definition with name, description, and input schema.
        """
        pass
    
    @abstractmethod
    def can_handle(self, tool_name: str) -> bool:
        """Check if this handler can handle the given tool name.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            True if this handler can handle the tool, False otherwise.
        """
        pass
    
    @abstractmethod
    async def handle(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Handle the tool call.
        
        Args:
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.
            
        Returns:
            List of text content responses.
        """
        pass
    
    async def handle_request(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Handle request using chain of responsibility pattern.
        
        Args:
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.
            
        Returns:
            List of text content responses.
            
        Raises:
            ValueError: If no handler in the chain can handle the tool.
        """
        if self.can_handle(tool_name):
            return await self.handle(tool_name, arguments)
        elif self._next_handler:
            return await self._next_handler.handle_request(tool_name, arguments)
        else:
            raise ValueError(f"No handler found for tool: {tool_name}")