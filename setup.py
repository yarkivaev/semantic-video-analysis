from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic-video-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Extract semantic descriptions from video files using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semantic-video-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "moviepy>=2.0.0",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=8.0.0",
        "sentence-transformers>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "openai-whisper @ git+https://github.com/openai/whisper.git",
        "numpy>=1.21.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.0",
        "ultralytics>=8.0.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "mcp": [
            "mcp>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "semantic-video-analysis=semantic_video_analysis.cli:main",
        ],
    },
)