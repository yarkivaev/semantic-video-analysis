"""Semantic Video Analysis Package

A tool for extracting semantic descriptions from video files using
computer vision and natural language processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .media_context import MediaAnalysis, MediaContext, Action
from .strategies import FrameSelectionAnalysis, PeriodicSelectionStrategy, FrameInfo

__all__ = ["MediaAnalysis", "MediaContext", "Action", "FrameSelectionAnalysis", "PeriodicSelectionStrategy", "FrameInfo"]