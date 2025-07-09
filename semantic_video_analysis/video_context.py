"""Video context analysis module for semantic video understanding.

This module provides data structures for representing and analyzing video content
through temporal actions and their contexts. It enables semantic understanding of
video segments by tracking actions over time intervals with associated content
descriptions.

Classes:
    Action: Represents a single action or event within a video segment.
    VideoContext: Container for a sequence of actions that form a video's context.
    VideoAnalysisModel: Model conducting video context analysis.
"""

from datetime import datetime


class Action:
    """Represents a single action or event within a video segment.
    
    An Action captures a discrete event or activity that occurs within a specific
    time interval in a video. It stores temporal boundaries and semantic content
    description for video analysis purposes.
    
    Attributes:
        start: The start time of the action.
        end: The end time of the action.
        content: Semantic description or metadata about the action.
    """
    
    def __init__(
        self,
        start,
        end,
        content
    ):
        """Initialize an Action with temporal boundaries and content.
        
        Args:
            start: The start time of the action.
            end: The end time of the action.
            content: Description or metadata about what happens during this action.
        """
        self.start = start
        self.end = end
        self.content = content

class VideoContext:
    """Container for a sequence of actions that form a video's semantic context.
    
    VideoContext aggregates multiple Action instances to represent the complete
    semantic understanding of a video. It provides a structured way to store and
    access the temporal sequence of events detected in video analysis.
    
    Attributes:
        actions: Unordered sequence of actions in the video.
    """

    def __init__(
        self,
        actions
    ):
        """Initialize VideoContext with a sequence of actions.
        
        Args:
            action: List of Action instances representing the video's temporal 
                events.
        """
        self.actions = actions

class VideoAnalysisModel:
    """Model conducting video context analysis.
    
    VideoAnalyzeModel encapsulates the results of video analysis, providing
    a standardized interface for accessing analyzed video context. This class
    serves as the output format for video analysis pipelines.
    
    Attributes:
        video_context: The analyzed video context containing
            detected actions and their temporal relationships.
    """
    
    def __init__(
        self,
        video_context
    ):
        """Initialize VideoAnalysysModel with analyzed video context.
        
        Args:
            video_context: The analyzed video context containing the sequence 
                of detected actions.
        """
        self.video_context = video_context
