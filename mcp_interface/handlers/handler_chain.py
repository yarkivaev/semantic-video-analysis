"""Handler chain dispatcher for MCP tools."""

from typing import Any, List, Tuple

import mcp.types as types
from .base_handler import BaseHandler


class HandlerChain:
    """Immutable chain that manages and dispatches tool calls through handlers."""
    
    def __init__(self, handlers: Tuple[BaseHandler, ...] = ()):
        """Initialize the handler chain.
        
        Args:
            handlers: Tuple of handlers to include in the chain.
        """
        self._handlers = tuple(handlers)
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the chain of responsibility between handlers."""
        # Create copies of handlers to avoid modifying the originals
        for i in range(len(self._handlers) - 1):
            self._handlers[i].set_next(self._handlers[i + 1])
    
    def with_handler(self, handler: BaseHandler) -> 'HandlerChain':
        """Create a new HandlerChain with an additional handler.
        
        Args:
            handler: Handler to add to the new chain.
            
        Returns:
            New HandlerChain instance with the added handler.
        """
        new_handlers = self._handlers + (handler,)
        return HandlerChain(new_handlers)
    
    def with_handlers(self, *handlers: BaseHandler) -> 'HandlerChain':
        """Create a new HandlerChain with multiple additional handlers.
        
        Args:
            handlers: Handlers to add to the new chain.
            
        Returns:
            New HandlerChain instance with the added handlers.
        """
        new_handlers = self._handlers + handlers
        return HandlerChain(new_handlers)
    
    @classmethod
    def of(cls, *handlers: BaseHandler) -> 'HandlerChain':
        """Create a new HandlerChain with the given handlers.
        
        Args:
            handlers: Handlers to include in the chain.
            
        Returns:
            New HandlerChain instance.
        """
        return cls(handlers)
    
    def get_all_tools(self) -> List[types.Tool]:
        """Get tool definitions from all handlers in the chain.
        
        Returns:
            List of all tool definitions.
        """
        tools = []
        for handler in self._handlers:
            tools.append(handler.get_tool_definition())
        return tools
    
    async def dispatch(self, tool_name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        """Dispatch a tool call to the appropriate handler.
        
        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.
            
        Returns:
            Response from the handler.
            
        Raises:
            ValueError: If no handler can handle the tool.
        """
        if not self._handlers:
            raise ValueError("No handlers registered in the chain")
        
        # Start the chain with the first handler
        return await self._handlers[0].handle_request(tool_name, arguments)