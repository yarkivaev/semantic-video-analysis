import argparse
import sys
import os
from .analyzer import VideoAnalyzer


def main():
    """Command-line interface for semantic video analysis."""
    parser = argparse.ArgumentParser(
        description="Extract semantic descriptions from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  semantic-video-analysis init                    # Download and cache the default model
  semantic-video-analysis init --model Salesforce/blip-image-captioning-large
  semantic-video-analysis analyze video.mp4
  semantic-video-analysis analyze video1.mp4 video2.mp4 video3.mp4
  semantic-video-analysis analyze *.mp4 --frames 10 --device cuda
  semantic-video-analysis analyze video.mp4 --output-dir results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize and download models')
    init_parser.add_argument(
        '--model',
        default='Salesforce/blip-image-captioning-base',
        help='BLIP model to download and cache (default: Salesforce/blip-image-captioning-base)'
    )
    init_parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='Device to test the model on (default: auto-detect)'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze video files')
    analyze_parser.add_argument(
        'videos',
        nargs='+',
        help='Path(s) to video files to analyze'
    )
    analyze_parser.add_argument(
        '--frames',
        type=int,
        default=5,
        help='Number of frames to extract from each video (default: 5)'
    )
    analyze_parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='Device to run the model on (default: auto-detect)'
    )
    analyze_parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory to save output JSON files (default: current directory)'
    )
    analyze_parser.add_argument(
        '--model',
        default='Salesforce/blip-image-captioning-base',
        help='BLIP model to use for captioning (default: Salesforce/blip-image-captioning-base)'
    )
    
    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'init':
        # Initialize the model to download and cache it
        print(f"Initializing video analyzer...")
        print(f"Device: {args.device or 'auto-detect'}")
        print(f"Model: {args.model}")
        print()
        print("Downloading and caching model components...")
        
        try:
            # Create analyzer instance which will download the model
            analyzer = VideoAnalyzer(device=args.device, model_name=args.model)
            
            # Test the model with a simple operation
            print("\nTesting model initialization...")
            
            # Create a small test image to verify model works
            import numpy as np
            from PIL import Image
            import tempfile
            
            # Create a simple test image
            test_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
                test_image.save(tmp.name)
                caption = analyzer.generate_caption(tmp.name)
                
            print(f"Model test successful! Generated caption: {caption}")
            print("\nInitialization complete! The model is now cached and ready for use.")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            sys.exit(1)
    
    elif args.command == 'analyze':
        # Validate video files exist
        for video_path in args.videos:
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                sys.exit(1)
        
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Initialize analyzer
        print(f"Initializing video analyzer...")
        print(f"Device: {args.device or 'auto-detect'}")
        print(f"Model: {args.model}")
        print(f"Frames per video: {args.frames}")
        print()
        
        try:
            analyzer = VideoAnalyzer(device=args.device, model_name=args.model)
            
            # Analyze videos
            descriptions = analyzer.analyze_videos(
                args.videos,
                output_dir=args.output_dir
            )
            
            print(f"\nAnalysis complete!")
            print(f"Processed {len(descriptions)} video(s)")
            print(f"Results saved to: {os.path.abspath(args.output_dir)}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()