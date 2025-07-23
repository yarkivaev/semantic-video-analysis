"""Media context analysis module for semantic media understanding.

This module provides data structures for representing and analyzing media content
through temporal actions and their contexts. It enables semantic understanding of
media segments by tracking actions over time intervals with associated content
descriptions. Supports various media types including video, audio, and images.

Classes:
    Action: Represents a single action or event within a media segment.
    MediaContext: Container for a sequence of actions that form a media's context.
    MediaAnalysis: Interface for conducting media context analysis.
"""

from datetime import datetime
from abc import ABC, abstractmethod


class Action:
    """Represents a single action or event within a media segment.
    
    An Action captures a discrete event or activity that occurs within a specific
    time interval in media content. It stores temporal boundaries and semantic content
    description for media analysis purposes.
    
    Attributes:
        start: The start time of the action.
        end: The end time of the action.
        content: Semantic description or metadata about the action.
        technical_analysis: Optional technical analysis data for the frame/segment.
    """
    
    def __init__(
        self,
        start,
        end,
        content,
        technical_analysis=None
    ):
        """Initialize an Action with temporal boundaries and content.
        
        Args:
            start: The start time of the action.
            end: The end time of the action.
            content: Description or metadata about what happens during this action.
            technical_analysis: Optional dict containing technical analysis metrics.
        """
        self.start = start
        self.end = end
        self.content = content
        self.technical_analysis = technical_analysis

class MediaContext:
    """Container for a sequence of actions that form a media's semantic context.
    
    MediaContext aggregates multiple Action instances to represent the complete
    semantic understanding of media content. It provides a structured way to store and
    access the temporal sequence of events detected in media analysis.
    
    Attributes:
        actions: Unordered sequence of actions in the media.
    """

    def __init__(
        self,
        actions
    ):
        """Initialize MediaContext with a sequence of actions.
        
        Args:
            actions: List of Action instances representing the media's temporal 
                events.
        """
        self.actions = actions

class MediaAnalysis(ABC):
    """Abstract media analysis interface that exposes analysis functionality.
    
    MediaAnalysis provides a standardized interface for conducting media analysis,
    exposing a single analyse method that returns analyzed media context.
    """
    
    @abstractmethod
    def analyse(self):
        """Analyze media content and return MediaContext.
        
        Returns:
            MediaContext: The analyzed media context containing detected actions.
        """
        pass
