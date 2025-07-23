import os
from moviepy import VideoFileClip

from ...media_context import Action, MediaContext, MediaAnalysis
from ...models.technical_analyzer import create_technical_analyzer


class FrameSelectionAnalysis(MediaAnalysis):
    """Analyzes videos by selecting specific frames and creating temporal actions."""
    
    def __init__(self, video_path, frame_analysis_fn, frame_selection_strategy, 
                 enable_technical_analysis=True):
        """Initialize with video path, frame analysis function and selection strategy.
        
        Args:
            video_path: Path to the video file
            frame_analysis_fn: Function to analyze frames (e.g., BLIP captioning)
            frame_selection_strategy: Strategy for selecting frames
            enable_technical_analysis: Whether to perform technical analysis on frames
        """
        self.video_path = video_path
        self.frame_analysis_fn = frame_analysis_fn
        self.frame_selection_strategy = frame_selection_strategy
        self.enable_technical_analysis = enable_technical_analysis
        self.technical_analyzer = None
        
        if enable_technical_analysis:
            self.technical_analyzer = create_technical_analyzer()
    
    def analyse(self):
        """Analyze video by selecting frames and creating MediaContext."""
        import cv2
        
        def extract_frame(video_path, frame_index, output_path):
            """Extract a specific frame from video."""
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(output_path, frame)
            cap.release()
            return ret
        
        strategy = self.frame_selection_strategy
            
        # Get video duration
        with VideoFileClip(self.video_path) as clip:
            duration = clip.duration
        
        # Select and analyze frames
        selected_frames = strategy.select_frames()
        base_name = os.path.basename(self.video_path).split('.')[0]
        os.makedirs("extracted_frames", exist_ok=True)
        
        frame_analyses = []
        for i, frame_info in enumerate(selected_frames):
            frame_path = f"extracted_frames/{base_name}_frame{i}.jpg"
            if extract_frame(self.video_path, frame_info.index, frame_path):
                # Semantic analysis (BLIP captioning)
                caption = self.frame_analysis_fn(frame_path)
                
                # Technical analysis (if enabled)
                technical_analysis = None
                if self.enable_technical_analysis and self.technical_analyzer:
                    # Load the frame for technical analysis
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        technical_analysis = self.technical_analyzer.analyze_frame(frame)
                
                frame_analyses.append((frame_info, caption, technical_analysis))
        
        # Create temporal actions
        actions = []
        for i, (frame_info, caption, technical_analysis) in enumerate(frame_analyses):
            # Calculate time boundaries
            start = 0 if i == 0 else (frame_analyses[i-1][0].timestamp + frame_info.timestamp) / 2
            end = duration if i == len(frame_analyses) - 1 else (frame_info.timestamp + frame_analyses[i+1][0].timestamp) / 2
            
            actions.append(Action(
                start=start,
                end=end,
                content={
                    "description": caption,
                    "frame_index": frame_info.index,
                    "frame_timestamp": frame_info.timestamp
                },
                technical_analysis=technical_analysis
            ))
        
        return MediaContext(actions=actions)