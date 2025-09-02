#!/usr/bin/env python3
"""Initialize and cache AI models at runtime."""

import os
import sys
import gc
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_model_cached(model_name: str, cache_type: str = "huggingface") -> bool:
    """Check if a model is already cached."""
    if cache_type == "huggingface":
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Check for model directory patterns
        model_id = model_name.replace("/", "--")
        if cache_dir.exists():
            for item in cache_dir.iterdir():
                if model_id.lower() in item.name.lower():
                    return True
    elif cache_type == "whisper":
        cache_dir = Path.home() / ".cache" / "whisper"
        if cache_dir.exists():
            # Whisper models are stored as files like large-v3.pt
            model_file = cache_dir / f"{model_name}.pt"
            return model_file.exists()
    return False


def init_blip_model():
    """Initialize BLIP model if not cached."""
    model_name = "Salesforce/blip-image-captioning-large"
    
    if check_model_cached(model_name, "huggingface"):
        logger.info("✅ BLIP model already cached, skipping download")
        return True
    
    logger.info("📦 Downloading BLIP model (first time only)...")
    logger.info("   This will take a few minutes...")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # Download and cache the model
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Clean up to free memory
        del processor
        del model
        gc.collect()
        
        logger.info("✅ BLIP model downloaded and cached successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download BLIP model: {e}")
        return False


def init_whisper_model():
    """Initialize Whisper model if not cached."""
    model_name = "base"
    
    if check_model_cached(model_name, "whisper"):
        logger.info("✅ Whisper model already cached, skipping download")
        return True
    
    logger.info("📦 Downloading Whisper base model (first time only)...")
    logger.info("   This is a small model (74MB), should be quick...")
    
    try:
        import whisper
        
        # Download and cache the model
        model = whisper.load_model(model_name)
        
        # Clean up to free memory
        del model
        gc.collect()
        
        logger.info("✅ Whisper model downloaded and cached successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download Whisper model: {e}")
        return False


def main():
    """Initialize all required models."""
    logger.info("=" * 60)
    logger.info("🚀 Initializing AI models for semantic video analysis")
    logger.info("=" * 60)
    
    # Check cache directories
    cache_info = []
    hf_cache = Path.home() / ".cache" / "huggingface"
    whisper_cache = Path.home() / ".cache" / "whisper"
    
    if hf_cache.exists():
        size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file()) / (1024**3)
        cache_info.append(f"  HuggingFace cache: {size:.2f} GB")
    
    if whisper_cache.exists():
        size = sum(f.stat().st_size for f in whisper_cache.rglob('*') if f.is_file()) / (1024**3)
        cache_info.append(f"  Whisper cache: {size:.2f} GB")
    
    if cache_info:
        logger.info("📊 Current cache status:")
        for info in cache_info:
            logger.info(info)
    
    # Initialize models
    success = True
    
    # Initialize BLIP
    if not init_blip_model():
        success = False
    
    # Initialize Whisper
    if not init_whisper_model():
        success = False
    
    if success:
        logger.info("=" * 60)
        logger.info("✨ All models initialized successfully!")
        logger.info("   Models are cached and will load instantly next time")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("❌ Some models failed to initialize")
        logger.error("   Please check the errors above")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())