import cv2
import numpy as np
import os
import json
import shutil
import tempfile
from scenedetect import VideoManager, SceneManager, ContentDetector
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from ultralytics import YOLO
from skimage.feature import hog
import joblib

warnings.filterwarnings('ignore')

# Global variables
temp_dir = None
cap = None
fps = None
frame_count = None
width = None
height = None
video_path = None

def initialize_video(video_path_input):
    """Initialize video properties and temporary directory."""
    global temp_dir, cap, fps, frame_count, width, height, video_path
    video_path = video_path_input
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_dir = "videos"
    os.makedirs(temp_dir, exist_ok=True)

def cleanup_temp_files():
    """Clean up temporary files."""
    global temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    temp_dir = tempfile.mkdtemp()

def detect_scenes(video_path_input):
    """Detect scenes using PySceneDetect."""
    video_manager = VideoManager([video_path_input])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0, min_scene_len=5))
    
    video_manager.set_duration()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return scene_list

def extract_scene_video(scene_list, scene_index=0):
    """Extract a specific scene for further processing."""
    global video_path, fps, width, height, temp_dir
    if scene_index >= len(scene_list):
        scene_index = 0
    
    scene = scene_list[scene_index]
    start_time = scene[0].get_seconds()
    end_time = scene[1].get_seconds()
    
    output_path = os.path.join(temp_dir, f'scene_{scene_index}.mp4')
    
    cap_local = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap_local.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    while cap_local.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
        ret, frame = cap_local.read()
        if not ret:
            break
        out.write(frame)
    
    cap_local.release()
    out.release()
    return output_path

def calculate_metric(frame, metric_type):
    """Calculate various metrics for a frame."""
    if metric_type == "blur":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    elif metric_type == "contrast":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.std()
    elif metric_type == "brightness":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean()
    elif metric_type == "color_temperature":
        b, g, r = cv2.split(frame)
        return (r.mean() - b.mean()) / (r.mean() + b.mean() + 1e-8)
    elif metric_type == "vignetting":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        center_brightness = gray[h//4:3*h//4, w//4:3*w//4].mean()
        corner_brightness = (gray[:h//4, :w//4].mean() + 
                           gray[:h//4, 3*w//4:].mean() + 
                           gray[3*h//4:, :w//4].mean() + 
                           gray[3*h//4:, 3*w//4:].mean()) / 4
        return (center_brightness - corner_brightness) / (center_brightness + 1e-8)
    elif metric_type == "saturation":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv[:,:,1].mean()
    elif metric_type == "sharpness":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3).var()
    return None

def analyze_video_metric(video_path_input, metric_type):
    """Analyze a metric for the entire video."""
    cap_local = cv2.VideoCapture(video_path_input)
    metrics = []
    
    while True:
        ret, frame = cap_local.read()
        if not ret:
            break
        metric_value = calculate_metric(frame, metric_type)
        metrics.append(metric_value)
    
    cap_local.release()
    return np.mean(metrics), np.std(metrics)

def calculate_shake_metric(video_path):
    """Calculate a scalar shake metric based on optical flow."""
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    max_corners = 100
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    shake_values = []
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        prev_points = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        if prev_points is None:
            prev_points = np.array([]).reshape(0, 1, 2)
        
        if len(prev_points) > 0:
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)
            if curr_points is not None and status is not None:
                good_points = status.ravel() == 1
                prev_points = prev_points[good_points]
                curr_points = curr_points[good_points]
                if len(prev_points) > 0:
                    displacements = np.sqrt(np.sum((curr_points - prev_points)**2, axis=2))
                    shake_values.append(np.mean(displacements))
        
        prev_gray = curr_gray
    
    cap.release()
    return np.mean(shake_values) if shake_values else 0.0

def calculate_color_balance_metric(video_path):
    """Calculate a scalar color balance metric based on RGB channel differences."""
    cap = cv2.VideoCapture(video_path)
    balance_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mean_rgb = np.mean(frame, axis=(0, 1))
        brightness = np.mean(mean_rgb)
        if brightness > 0:
            balance = (abs(mean_rgb[2] - mean_rgb[1]) + abs(mean_rgb[1] - mean_rgb[0]) + abs(mean_rgb[0] - mean_rgb[2])) / brightness
            balance_values.append(balance)
    
    cap.release()
    return np.mean(balance_values) if balance_values else 0.0

def calculate_light_hardness_metric(video_path):
    """Calculate a scalar light hardness metric based on contrast."""
    cap = cv2.VideoCapture(video_path)
    contrast_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        contrast_values.append(contrast)
    
    cap.release()
    return np.mean(contrast_values) if contrast_values else 0.0

def calculate_depth_of_field_metric(video_path):
    """Calculate a scalar depth of field metric based on sharpness."""
    cap = cv2.VideoCapture(video_path)
    sharpness_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3).var()
        sharpness_values.append(sharpness)
    
    cap.release()
    return np.mean(sharpness_values) if sharpness_values else 0.0

def get_vignetting_form_frame(frame):
    """Calculate vignetting parameters from a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inv_gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (x_center, y_center), (major_axis, minor_axis), angle = ellipse
            if major_axis > frame.shape[0]*0.5:
                return x_center, y_center, major_axis, minor_axis, angle
    return None

def get_vignetting_form_video(video_path_input):
    """Calculate average vignetting parameters for a video."""
    x_center, y_center, major_axis, minor_axis, angle = 0, 0, 0, 0, 0
    not_v_counter = 0
    proc_frames_counter = 0
    
    cap_local = cv2.VideoCapture(video_path_input)
    while True:
        ret, frame = cap_local.read()
        if not ret:
            break
        proc_frames_counter += 1
        new_info = get_vignetting_form_frame(frame)
        
        if new_info:
            new_x_center, new_y_center, new_major_axis, new_minor_axis, new_angle = new_info
            x_center += new_x_center
            y_center += new_y_center
            major_axis += new_major_axis
            minor_axis += new_minor_axis
            angle += new_angle
        else:
            not_v_counter += 1
    
    cap_local.release()
    
    if not_v_counter/proc_frames_counter > 0.3:
        return None
    else:
        x_center /= proc_frames_counter
        y_center /= proc_frames_counter
        major_axis /= proc_frames_counter
        minor_axis /= proc_frames_counter
        angle /= proc_frames_counter
        return {
            "center_x": x_center,
            "center_y": y_center,
            "major_axis": major_axis/2,
            "minor_axis": minor_axis/2,
            "angle": angle
        }

def calculate_patch_sharpness(gray, patch_size, x, y):
    """Calculate sharpness of a patch using Sobel variance."""
    patch = gray[y:y+patch_size, x:x+patch_size]
    if patch.size == 0:
        return 0.0
    return cv2.Sobel(patch, cv2.CV_64F, 1, 1, ksize=3).var()

def calculate_background_blur_metric(video_path, patch_size=100, grid_size=5, sharpness_threshold=0.1):
    """Calculate a scalar background blur metric."""
    cap = cv2.VideoCapture(video_path)
    frame_blur_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        step = int(patch_size)
        
        sharpness_values = []
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                sharpness = calculate_patch_sharpness(gray, patch_size, x, y)
                sharpness_values.append((sharpness, x, y))
        
        if not sharpness_values:
            continue
        
        max_sharpness = max(s[0] for s in sharpness_values)
        if max_sharpness == 0:
            continue
        sharp_patches = [(s, x, y) for s, x, y in sharpness_values if s >= max_sharpness * (1 - sharpness_threshold)]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        for _, x, y in sharp_patches:
            cv2.rectangle(mask, (x, y), (x + patch_size, y + patch_size), 255, -1)
        
        background = gray.copy()
        background[mask != 0] = 0
        
        if background.size > 0:
            blur = cv2.Sobel(background, cv2.CV_64F, 1, 1, ksize=3).var()
            frame_blur_values.append(blur)
    
    cap.release()
    return np.mean(frame_blur_values) if frame_blur_values else 0.0

def process_video_segmented_objects(model, video_path, frame_step=5, top=5):
    """Calculate the average area of top objects in segmented video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tops_sums_area = 0
    proc_count_frames = 0
    frame_count = 0
    
    for _ in tqdm(range(total_frames), total=total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            areas = []
            results = model(frame, verbose=False)
            for result in results:
                if result.masks is not None:
                    for mask in result.masks.data:
                        area = np.mean(mask.cpu().numpy() > 0)
                        areas.append(area)
            top_areas = sorted(areas, reverse=True)[:top]
            tops_sums_area += sum(top_areas)
            proc_count_frames += 1
        frame_count += 1
    
    cap.release()
    tops_sums_area = tops_sums_area/proc_count_frames if proc_count_frames > 0 else 0
    return tops_sums_area

def relative_area_to_plane_class(relative_area):
    """Classify shot type based on relative area."""
    if relative_area < 0.05:
        return "Дальний план"
    elif 0.05 <= relative_area < 0.15:
        return "Общий план"
    elif 0.15 <= relative_area < 0.40:
        return "Средний план"
    else:
        return "Крупный план"

def extract_hog_features(image):
    """Extract HOG features from an image."""
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(image, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=True)
    return features

def extract_sift_features(image):
    """Extract SIFT features from an image."""
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)

def extract_orb_features(image):
    """Extract ORB features from an image."""
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(32)
    return np.mean(descriptors, axis=0)

def predict_shot_type(video_path, svm_model_path='svm_model.joblib'):
    """Predict shot type using SVM model."""
    svm = joblib.load(svm_model_path)
    cap = cv2.VideoCapture(video_path)
    predictions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hog_features = extract_hog_features(frame)
        sift_features = extract_sift_features(frame)
        orb_features = extract_orb_features(frame)
        features = np.concatenate([hog_features, sift_features, orb_features])
        prediction = svm.predict([features])[0]
        predictions.append(prediction)
    
    cap.release()
    
    # Return the most common prediction
    if predictions:
        return max(set(predictions), key=predictions.count)
    return 0  # Default to normal if no predictions

def analyze_video(video_path_input, svm_model_path='svm_model.joblib'):
    """Analyze video scenes and collect characteristics."""
    initialize_video(video_path_input)
    scene_list = detect_scenes(video_path_input)
    model = YOLO('yolo11n-seg.pt')
    
    results = []
    
    for scene_index in range(len(scene_list)):
        scene_path = extract_scene_video(scene_list, scene_index)
        
        # Predict shot type using SVM
        shot_type_svm = predict_shot_type(scene_path, svm_model_path)
        shot_type_label = {0: "Normal", 1: "Fisheye", 2: "Wide-angle"}[shot_type_svm]
        
        # Collect metrics
        blur_mean, blur_std = analyze_video_metric(scene_path, "blur")
        contrast_mean, contrast_std = analyze_video_metric(scene_path, "contrast")
        brightness_mean, brightness_std = analyze_video_metric(scene_path, "brightness")
        color_temp = analyze_video_metric(scene_path, "color_temperature")[0]
        saturation = analyze_video_metric(scene_path, "saturation")[0]
        sharpness_mean, sharpness_std = analyze_video_metric(scene_path, "sharpness")
        shake = calculate_shake_metric(scene_path)
        color_balance = calculate_color_balance_metric(scene_path)
        light_hardness = calculate_light_hardness_metric(scene_path)
        depth_of_field = calculate_depth_of_field_metric(scene_path)
        vignetting_info = get_vignetting_form_video(scene_path)
        background_blur = calculate_background_blur_metric(scene_path)
        relative_area = process_video_segmented_objects(model, scene_path)
        shot_type_area = relative_area_to_plane_class(relative_area)
        
        scene_data = {
            "scene_index": scene_index,
            "start_time": scene_list[scene_index][0].get_seconds(),
            "end_time": scene_list[scene_index][1].get_seconds(),
            "metrics": {
                "blur": {"mean": float(blur_mean), "std": float(blur_std)},
                "contrast": {"mean": float(contrast_mean), "std": float(contrast_std)},
                "brightness": {"mean": float(brightness_mean), "std": float(brightness_std)},
                "color_temperature": float(color_temp),
                "saturation": float(saturation),
                "sharpness": {"mean": float(sharpness_mean), "std": float(sharpness_std)},
                "camera_shake": float(shake),
                "color_balance": float(color_balance),
                "light_hardness": float(light_hardness),
                "depth_of_field": float(depth_of_field),
                "vignetting": vignetting_info if vignetting_info else "No vignetting detected",
                "background_blur": float(background_blur),
                "relative_area": float(relative_area),
                "shot_type_area": shot_type_area,
                "shot_type_svm": shot_type_label
            }
        }
        results.append(scene_data)
    
    cleanup_temp_files()
    
    return {"video_path": video_path_input, "scenes": results}