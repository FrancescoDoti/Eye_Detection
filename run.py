import os
import cv2
import time
import torch
import torchvision
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from object_detector import ObjectDetector
from gaze_tracker import GazeTracker
from gaze_driven_detection import GazeDrivenObjectDetection
from user_profile import UserProfile
from context_aware_detection import ContextAwareObjectDetection
from evaluation import evaluate_real_time_performance, run_user_study
from main import main, main_enhanced


def test_with_dataset(image_folder, gaze_data_file=None):
    """
    Test the system with a dataset of images and optional gaze data
    
    Args:
        image_folder (str): Path to folder containing images
        gaze_data_file (str): Path to file containing gaze data (optional)
    """
    system = GazeDrivenObjectDetection(confidence_threshold=0.5, gaze_influence=0.7)
    
    # Load images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    # Load gaze data if available
    gaze_data = {}
    if gaze_data_file and os.path.exists(gaze_data_file):
        import json
        with open(gaze_data_file, 'r') as f:
            gaze_data = json.load(f)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # Set gaze position if available in gaze data
        if image_file in gaze_data:
            gaze_x, gaze_y = gaze_data[image_file]
            system.gaze_tracker.gaze_position = (gaze_x, gaze_y)
        
        # Process image
        result = system.process_frame(image)
        
        # Display result
        cv2.imshow("Dataset Testing", result)
        key = cv2.waitKey(0)  # Wait until key press
        
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()


