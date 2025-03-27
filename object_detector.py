import cv2
import numpy as np

class ObjectDetectorCV:
    def __init__(self, min_area=500, nms_threshold=0.3, face_scaleFactor=1.1, face_minNeighbors=5):
        """
        Initialize the object detector without deep learning.
        
        Args:
            min_area (int): Minimum area in pixels to consider a contour as a valid detection.
            nms_threshold (float): Overlap threshold for non-maximum suppression.
            face_scaleFactor (float): Scale factor for Haar cascade face detection.
            face_minNeighbors (int): Min neighbors for Haar cascade face detection.
        """
        self.min_area = min_area
        self.nms_threshold = nms_threshold
        
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_scaleFactor = face_scaleFactor
        self.face_minNeighbors = face_minNeighbors

    def detect(self, image):
        """
        Detect objects in the given image using classical image processing techniques,
        while masking out face and hair regions.
        
        Args:
            image (numpy.ndarray): Input image in BGR format (as read by OpenCV)
            
        Returns:
            list: List of detections, each containing 
                  [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.face_scaleFactor,
            minNeighbors=self.face_minNeighbors,
            minSize=(30, 30)
        )
        # Create a mask for faces and hair. Instead of masking just the face,
        # extend the masked area upward (using a fraction of the face height) to cover hair.
        face_mask = np.ones(gray.shape, dtype="uint8") * 255
        for (fx, fy, fw, fh) in faces:
            pad = 10
            # Define extra padding above the face to cover hair.
            hair_extra = int(0.5 * fh)
            cv2.rectangle(face_mask, 
                          (max(fx - pad, 0), max(fy - hair_extra - pad, 0)), 
                          (min(fx + fw + pad, gray.shape[1]), min(fy + fh + pad, gray.shape[0])), 
                          0, thickness=-1)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use a threshold to binarize the image (invert so objects are white)
        ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Mask out face and hair regions from the thresholded image
        thresh = cv2.bitwise_and(thresh, thresh, mask=face_mask)
        
        # Apply morphological operations to improve segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours in the processed image
        contours, _ = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue  # Skip small contours
            
            # Approximate contour to reduce irregularities
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Get the bounding box of the approximated contour
            x, y, w, h = cv2.boundingRect(approx)
            
            # Filter out extreme aspect ratios (e.g., very long and thin shapes)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Calculate solidity (ratio of contour area to its convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < 0.5:
                    continue
            
            # If passed all filters, add the detection with dummy confidence
            confidence = 1.0
            class_id = 0
            class_name = "object"
            detections.append([x, y, x + w, y + h, confidence, class_id, class_name])
        
        # Apply non-maximum suppression to reduce overlapping boxes
        detections = self.non_max_suppression(detections, self.nms_threshold)
        return detections

    def non_max_suppression(self, detections, overlap_thresh=0.3):
        """
        Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes.
        
        Args:
            detections (list): List of detections in the format 
                               [x1, y1, x2, y2, confidence, class_id, class_name].
            overlap_thresh (float): Overlap threshold above which boxes are suppressed.
            
        Returns:
            list: Filtered detections after applying NMS.
        """
        if len(detections) == 0:
            return []
        
        boxes = np.array([d[:5] for d in detections], dtype=float)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        idxs = np.argsort(scores)
        pick = []
        
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[last] + areas[idxs[:-1]] - inter)
            idxs = np.delete(
                idxs,
                np.concatenate(([len(idxs)-1], np.where(iou > overlap_thresh)[0]))
            )
        
        filtered_detections = [detections[i] for i in pick]
        return filtered_detections
