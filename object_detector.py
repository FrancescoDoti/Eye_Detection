import cv2
import time
import torch
import torchvision
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F


class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the object detector with a pre-trained model
        
        Args:
            confidence_threshold (float): Threshold for detection confidence
        """
        # Load a pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # Set to evaluation mode
        
        # Move the model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set confidence threshold for detections
        self.confidence_threshold = confidence_threshold
        
        # COCO dataset class names
        self.classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, image):
        """
        Detect objects in the given image
        
        Args:
            image (numpy.ndarray): Input image in BGR format (OpenCV)
            
        Returns:
            list: List of detections, each containing [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image_rgb)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract results
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Format detections
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            class_id = label
            class_name = self.classes[class_id]
            detections.append([x1, y1, x2, y2, score, class_id, class_name])
        
        return detections
