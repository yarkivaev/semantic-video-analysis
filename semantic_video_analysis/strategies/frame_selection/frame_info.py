from dataclasses import dataclass


@dataclass
class FrameInfo:
    """Information about a selected frame."""
    index: int
    timestamp: float