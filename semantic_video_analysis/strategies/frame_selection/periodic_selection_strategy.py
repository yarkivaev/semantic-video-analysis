from moviepy import VideoFileClip

from .frame_info import FrameInfo


class PeriodicSelectionStrategy:
    """Strategy for selecting frames from video with a specified period."""
    
    def __init__(self, period: float, duration: float, fps: float):
        """Initialize with video parameters.
        
        Args:
            period: Time period in seconds between selected frames.
            duration: Video duration in seconds.
            fps: Frames per second of the video.
        """
        self.period = period
        self.duration = duration
        self.fps = fps
    
    @classmethod
    def from_video_file(cls, video_path: str, period: float = 1.0):
        """Create strategy by reading video metadata.
        
        Args:
            video_path: Path to the video file.
            period: Time period in seconds between selected frames.
            
        Returns:
            PeriodicSelectionStrategy: Initialized strategy with video metadata.
        """
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            fps = clip.fps
        
        return cls(period, duration, fps)
    
    def select_frames(self):
        """Select frames at periodic intervals."""
        frames = []
        timestamp = 0
        
        while timestamp < self.duration:
            frame_index = int(timestamp * self.fps)
            frames.append(FrameInfo(index=frame_index, timestamp=timestamp))
            timestamp += self.period
            
        return frames