"""Technical frame analysis module for video quality assessment.

This module provides functionality to analyze individual video frames for various
technical characteristics including clarity, contrast, brightness, color properties,
sharpness, and lens type classification.
"""

import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from ultralytics import YOLO
import os
import warnings

warnings.filterwarnings('ignore')


class TechnicalFrameAnalyzer:
    """Analyzer for technical characteristics of individual video frames."""
    
    def __init__(self, svm_model_path=None, yolo_model_path=None):
        """Initialize the technical analyzer with optional model paths.
        
        Args:
            svm_model_path: Path to SVM model for lens type classification
            yolo_model_path: Path to YOLO model for object detection
        """
        self.svm_model = None
        self.yolo_model = None
        
        # Load SVM model if path provided
        if svm_model_path and os.path.exists(svm_model_path):
            try:
                self.svm_model = joblib.load(svm_model_path)
            except Exception as e:
                print(f"Warning: Could not load SVM model from {svm_model_path}: {e}")
        
        # Load YOLO model if path provided
        if yolo_model_path:
            try:
                self.yolo_model = YOLO(yolo_model_path)
            except Exception as e:
                print(f"Warning: Could not load YOLO model from {yolo_model_path}: {e}")
    
    def analyze_frame(self, frame):
        """Perform comprehensive technical analysis on a single frame.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            dict: Technical analysis results
        """
        if frame is None:
            return {}
        
        analysis = {}
        
        # Basic image quality metrics
        analysis['clarity'] = self._analyze_clarity(frame)
        analysis['contrast'] = self._analyze_contrast(frame)
        analysis['brightness'] = self._analyze_brightness(frame)
        analysis['sharpness'] = self._analyze_sharpness(frame)
        analysis['saturation'] = self._analyze_saturation(frame)
        
        # Color analysis
        color_balance, r, g, b = self._analyze_color_balance_and_distribution(frame)
        analysis['color_balance_metric'] = color_balance
        analysis['color_distribution'] = {'r': r, 'g': g, 'b': b}
        analysis['color_temperature'] = self._analyze_color_temperature(frame)
        
        # Advanced analysis (if models available)
        if self.svm_model is not None:
            analysis['lens_type'] = self._classify_lens_type(frame)
        
        if self.yolo_model is not None:
            analysis['main_objects_percentage'] = self._analyze_objects(frame)
            analysis['shot_type'] = self._assign_shot_type(analysis['main_objects_percentage'])
        
        # Add quality categories
        analysis.update(self._assign_quality_categories(analysis))
        
        return analysis
    
    def _analyze_clarity(self, frame):
        """Calculate frame clarity using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def _analyze_contrast(self, frame):
        """Calculate contrast using standard deviation of grayscale image."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    def _analyze_brightness(self, frame):
        """Calculate brightness as mean of grayscale image, normalized by 255."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean() / 255)
    
    def _analyze_sharpness(self, frame):
        """Calculate sharpness as variance of Sobel filter."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3).var())
    
    def _analyze_saturation(self, frame):
        """Calculate saturation as mean of HSV saturation channel."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(hsv[:,:,1].mean() / 255)
    
    def _analyze_color_balance_and_distribution(self, frame):
        """Calculate color balance and distribution."""
        mean_rgb = np.mean(frame, axis=(0, 1))  # [B, G, R]
        brightness = np.mean(mean_rgb)
        balance = 0.0
        if brightness > 0:
            balance = (abs(mean_rgb[2] - mean_rgb[1]) + 
                      abs(mean_rgb[1] - mean_rgb[0]) + 
                      abs(mean_rgb[0] - mean_rgb[2])) / brightness
        
        return (float(balance / 6), 
                float(mean_rgb[0] / 255), 
                float(mean_rgb[1] / 255), 
                float(mean_rgb[2] / 255))
    
    def _analyze_color_temperature(self, frame):
        """Calculate color temperature as (R mean - B mean) / (R mean + B mean)."""
        b, g, r = cv2.split(frame)
        return float((r.mean() - b.mean()) / (r.mean() + b.mean() + 1e-8))
    
    def _classify_lens_type(self, frame):
        """Classify lens type using SVM model with HOG, SIFT, and ORB features."""
        if self.svm_model is None:
            return None
        
        try:
            hog_features = self._extract_hog_features(frame)
            sift_features = self._extract_sift_features(frame)
            orb_features = self._extract_orb_features(frame)
            features = np.concatenate([hog_features, sift_features, orb_features])
            prediction = int(self.svm_model.predict([features])[0])
            return self._assign_lens_category(prediction)
        except Exception as e:
            print(f"Warning: Lens type classification failed: {e}")
            return None
    
    def _extract_hog_features(self, image):
        """Extract HOG features for SVM classification."""
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features, _ = hog(image, orientations=9, pixels_per_cell=(32, 32), 
                         cells_per_block=(2, 2), visualize=True)
        return features
    
    def _extract_sift_features(self, image):
        """Extract SIFT features for SVM classification."""
        image = cv2.resize(image, (512, 512))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.zeros(128)
        return np.mean(descriptors, axis=0)
    
    def _extract_orb_features(self, image):
        """Extract ORB features for SVM classification."""
        image = cv2.resize(image, (512, 512))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None:
            return np.zeros(32)
        return np.mean(descriptors, axis=0)
    
    def _analyze_objects(self, frame):
        """Calculate the relative area of main objects using YOLO."""
        if self.yolo_model is None:
            return None
        
        try:
            frame_resized = self._resize_for_yolo(frame)
            results = self.yolo_model(frame_resized, verbose=False)
            
            areas = []
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    areas.extend([np.mean(mask > 0) for mask in masks])
            
            # Get top 5 areas
            top_areas = sorted(areas, reverse=True)[:5]
            return float(sum(top_areas) * 100) if top_areas else 0.0
            
        except Exception as e:
            print(f"Warning: Object analysis failed: {e}")
            return None
    
    def _resize_for_yolo(self, frame, target_size=416):
        """Resize frame for YOLO processing."""
        h, w = frame.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def _assign_quality_categories(self, analysis):
        """Assign quality categories based on metric values."""
        categories = {}
        
        if 'clarity' in analysis:
            clarity = analysis['clarity']
            categories['clarity_category'] = ('good' if clarity > 400 
                                           else 'average' if clarity > 200 
                                           else 'bad')
        
        if 'contrast' in analysis:
            contrast = analysis['contrast']
            categories['contrast_category'] = ('good' if contrast > 40 
                                            else 'average' if contrast > 20 
                                            else 'bad')
        
        if 'brightness' in analysis:
            brightness = analysis['brightness']
            if 0.3 <= brightness <= 0.7:
                categories['brightness_category'] = 'good'
            elif 0.2 <= brightness < 0.3 or 0.7 < brightness <= 0.8:
                categories['brightness_category'] = 'average'
            else:
                categories['brightness_category'] = 'bad'
        
        if 'color_temperature' in analysis:
            temp = analysis['color_temperature']
            if -0.2 <= temp <= 0.2:
                categories['color_temperature_category'] = 'neutral'
            elif 0.2 < temp <= 0.3:
                categories['color_temperature_category'] = 'probably_warm'
            elif -0.3 <= temp < -0.2:
                categories['color_temperature_category'] = 'probably_cold'
            elif temp > 0.3:
                categories['color_temperature_category'] = 'warm'
            else:
                categories['color_temperature_category'] = 'cold'
        
        if 'saturation' in analysis:
            saturation = analysis['saturation']
            if 0.4 <= saturation <= 0.7:
                categories['saturation_category'] = 'good'
            elif 0.2 <= saturation < 0.4:
                categories['saturation_category'] = 'average'
            else:
                categories['saturation_category'] = 'bad'
        
        if 'sharpness' in analysis:
            sharpness = analysis['sharpness']
            categories['sharpness_category'] = ('good' if sharpness > 400 
                                             else 'average' if sharpness > 200 
                                             else 'bad')
        
        return categories
    
    def _assign_lens_category(self, lens_value):
        """Assign lens type category based on classification result."""
        lens_mapping = {
            0: 'standard lens',
            1: 'fisheye lens',
            2: 'wide-angle lens'
        }
        return lens_mapping.get(lens_value, 'unknown')
    
    def _assign_shot_type(self, objects_percentage):
        """Assign shot type based on main objects percentage."""
        if objects_percentage is None:
            return None
        
        if objects_percentage < 5:
            return 'long shot'
        elif objects_percentage < 15:
            return 'wide shot'
        elif objects_percentage < 40:
            return 'medium shot'
        else:
            return 'close-up'


def create_technical_analyzer(enable_svm=True, enable_yolo=True):
    """Factory function to create a TechnicalFrameAnalyzer with default model paths.
    
    Args:
        enable_svm: Whether to enable SVM model for lens classification
        enable_yolo: Whether to enable YOLO model for object detection
    
    Returns:
        TechnicalFrameAnalyzer: Configured analyzer instance
    """
    # Default model paths - can be overridden
    svm_path = None
    yolo_path = None
    
    if enable_svm:
        # Try to find SVM model in common locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_svm_paths = [
            os.path.join(current_dir, 'svm_model.joblib'),  # Same directory as this module
            'semantic_video_analysis/models/svm_model.joblib',
            'models/svm_model.joblib',
            'svm_model.joblib'
        ]
        for path in possible_svm_paths:
            if os.path.exists(path):
                svm_path = path
                break
    
    if enable_yolo:
        yolo_path = 'yolo11n-seg.pt'  # YOLO will auto-download if not found
    
    return TechnicalFrameAnalyzer(svm_model_path=svm_path, yolo_model_path=yolo_path)