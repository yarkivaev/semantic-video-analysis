# Semantic Video Analysis

A Python tool for extracting semantic descriptions from video files using computer vision and natural language processing. This tool uses the BLIP (Bootstrapping Language-Image Pre-training) model to generate captions for video frames and creates comprehensive JSON descriptions of video content.

## Features

- Extract key frames from videos at regular intervals
- Generate AI-powered captions for video frames using BLIP model
- Create structured JSON descriptions with rich metadata
- Detect objects, scenes, and estimate mood from video content
- Batch processing support via command-line interface
- Configurable device selection (CPU/CUDA)
- Customizable frame extraction count

## Installation

### From Source

```bash
git clone https://github.com/yourusername/semantic-video-analysis.git
cd semantic-video-analysis
pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Command Line Interface

Initialize the model (recommended before first use):
```bash
# Download and cache the default model
semantic-video-analysis init

# Download a specific model
semantic-video-analysis init --model Salesforce/blip-image-captioning-large
```

Basic usage:
```bash
semantic-video-analysis analyze video.mp4
```

Batch processing:
```bash
semantic-video-analysis analyze video1.mp4 video2.mp4 video3.mp4
```

Advanced options:
```bash
# Extract 10 frames per video using CUDA
semantic-video-analysis analyze *.mp4 --frames 10 --device cuda

# Save results to specific directory
semantic-video-analysis analyze video.mp4 --output-dir results/

# Use a different BLIP model
semantic-video-analysis analyze video.mp4 --model Salesforce/blip-image-captioning-large
```

### Python API

```python
from semantic_video_analysis import VideoAnalyzer

# Initialize analyzer
analyzer = VideoAnalyzer(device="cuda")  # or "cpu"

# Analyze single video
description = analyzer.describe_video("path/to/video.mp4", num_frames=5)
print(description)

# Analyze multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
descriptions = analyzer.analyze_videos(video_paths, output_dir="results/")
```

## Output Format

The tool generates JSON descriptions with the following structure:

```json
{
  "type": "video",
  "metadata": {
    "fileName": "example.mp4",
    "filePath": "/path/to/example.mp4",
    "fileSize": 1234567,
    "createdAt": "2024-06-24 10:30:00 UTC",
    "description": "Combined caption from first frames",
    "tags": ["object1", "object2", "action"]
  },
  "duration": 120.5,
  "resolution": {
    "width": 1920,
    "height": 1080
  },
  "frameRate": 30.0,
  "hasAudio": true,
  "videoFormat": "mp4",
  "contentAnalysis": {
    "contentOverview": "Full video content description",
    "actionIntroduction": "First frame caption",
    "timeBoundDetails": [
      {
        "detailStartTime": 0.0,
        "detailEndTime": 24.1,
        "detailDescription": "Frame caption at this time",
        "detailConfidence": 0.82
      }
    ],
    "detectedObjects": ["person", "car", "building"],
    "detectedScenes": ["outdoor", "indoor"],
    "estimatedMood": "neutral"
  }
}
```

## CLI Commands

### init - Initialize and download models

```
semantic-video-analysis init [OPTIONS]

Options:
  --model TEXT              BLIP model to download (default: Salesforce/blip-image-captioning-base)
  --device {cpu,cuda}       Device to test model on (default: auto-detect)
  --help                    Show help message
```

### analyze - Analyze video files

```
semantic-video-analysis analyze [OPTIONS] VIDEO_FILES...

Options:
  --frames INT              Number of frames to extract (default: 5)
  --device {cpu,cuda}       Device to run model on (default: auto-detect)
  --output-dir PATH         Output directory for JSON files (default: current)
  --model TEXT              BLIP model name (default: Salesforce/blip-image-captioning-base)
  --help                    Show help message
```

### Global Options

```
  --version                 Show version information
  --help                    Show help message
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- MoviePy 2.0+
- OpenCV 4.5+
- Pillow 8.0+
- sentence-transformers 2.0+
- accelerate 0.20+

## Model Information

This tool uses the BLIP (Bootstrapping Language-Image Pre-training) model from Salesforce for image captioning. The default model is `Salesforce/blip-image-captioning-base`, but you can use other BLIP variants:

- `Salesforce/blip-image-captioning-base` - Base model, good balance of speed and quality
- `Salesforce/blip-image-captioning-large` - Larger model, better quality but slower

## Performance Tips

1. **GPU Usage**: Use `--device cuda` for faster processing if you have a CUDA-capable GPU
2. **Frame Count**: Adjust `--frames` based on video length and desired detail level
3. **Batch Processing**: Process multiple videos in one command for efficiency
4. **Model Selection**: Use the base model for faster processing or large model for better quality

## Limitations

- The quality of descriptions depends on the BLIP model's training data
- Very long videos may take significant processing time
- Extracted frames are saved temporarily in an `extracted_frames` directory
- Scene and mood detection are basic and may be improved with specialized models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details