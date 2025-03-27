import cv2
import numpy as np

class ObjectDetectorCV:
    def __init__(self, 
                 min_area=100,  # Minimum contour area
                 nms_threshold=0.3):
        """
        Initialize the object detector with basic detection capabilities.
        
        Args:
            min_area (int): Minimum area in pixels to consider a contour as a valid detection.
            nms_threshold (float): Overlap threshold for non-maximum suppression.
        """
        self.min_area = min_area
        self.nms_threshold = nms_threshold

    def detect(self, image):
        """
        Detect objects in the given image using simple contour detection.
        
        Args:
            image (numpy.ndarray): Input image in BGR format (as read by OpenCV)
            
        Returns:
            list: List of detections, each containing 
                  [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken parts
        dilated = cv2.dilate(edges, None, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        img_h, img_w = image.shape[:2]
        for cnt in contours:
            # Area filtering
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Basic aspect ratio filtering
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            # Filter out detections that are too big relative to the image size
            if w > 0.5 * img_w or h > 0.5 * img_h:
                continue
            
            # Confidence calculation (as before, based on area)
            confidence = min(area / 1000.0, 1.0)
            
            # Add detection: [x1, y1, x2, y2, confidence, class_id, class_name]
            detections.append([x, y, x + w, y + h, confidence, 0, "object"])
        
        # Apply non-maximum suppression with our modified logic
        return self.non_max_suppression(detections, self.nms_threshold)

    def non_max_suppression(self, detections, overlap_thresh=0.3):
        """
        Apply Non-Maximum Suppression (NMS) to reduce overlapping detections.
        Modified such that when two detections overlap too much, the smaller one (by area)
        is kept.
        """
        if len(detections) == 0:
            return []
        
        # Convert detections to numpy array for easier processing.
        # We'll work with the first 5 columns: x1, y1, x2, y2, and confidence.
        boxes = np.array([d[:5] for d in detections], dtype=float)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute the area of each detection.
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort indices based on area in ascending order (smaller boxes first).
        idxs = np.argsort(areas)
        keep = []

        # Process detections: always keep the smallest one and remove overlapping larger ones.
        while len(idxs) > 0:
            # The detection with the smallest area
            i = idxs[0]
            keep.append(i)
            
            # Compute the intersection with the rest of the boxes.
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            # Compute width and height of the intersection boxes.
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            
            # Compute Intersection over Union (IoU)
            union = areas[i] + areas[idxs[1:]] - inter
            iou = inter / union
            
            # Identify indices where IoU is above the threshold
            # For these indices, we remove the larger detection (which are in idxs[1:]).
            # Since our list is sorted in ascending order by area, all boxes in idxs[1:] are larger than box i.
            remove_idxs = np.where(iou > overlap_thresh)[0]
            
            # Remove i (already processed) and any indices with too high overlap.
            idxs = np.delete(idxs, np.concatenate(([0], remove_idxs + 1)))
        
        # Return the filtered detections
        return [detections[i] for i in keep]