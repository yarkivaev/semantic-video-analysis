"""Semantic Video Analysis Package

A tool for extracting semantic descriptions from video files using
computer vision and natural language processing.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .analyzer import VideoAnalyzer
from .cli import main

__all__ = ["VideoAnalyzer", "main"]