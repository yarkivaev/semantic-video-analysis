import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipModel:
    """Resource wrapper for BLIP model and processor."""
    
    def __init__(self, device=None, model_name="3а"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.processor = None
        self.model = None
    
    def __enter__(self):
        """Load model and processor when entering context."""
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        os.makedirs("extracted_frames", exist_ok=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_caption(self, image):
        """Generate caption for image using loaded model."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs)
        
        return self.processor.decode(output[0], skip_special_tokens=True)