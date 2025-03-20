import cv2
import numpy as np

class ObjectDetectorCV:
    def __init__(self, min_area=500, nms_threshold=0.3):
        """
        Initialize the object detector without deep learning.
        
        Args:
            min_area (int): Minimum area in pixels to consider a contour as a valid detection.
            nms_threshold (float): Overlap threshold for non-maximum suppression.
        """
        self.min_area = min_area
        self.nms_threshold = nms_threshold

    def detect(self, image):
        """
        Detect objects in the given image using classical image processing techniques.
        
        Args:
            image (numpy.ndarray): Input image in BGR format (as read by OpenCV)
            
        Returns:
            list: List of detections, each containing 
                  [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise and help edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Perform Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        
        # Optionally dilate edges to close gaps
        dilated = cv2.dilate(edges, None, iterations=1)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            # Filter out small contours based on a minimum area to reduce noise
            if cv2.contourArea(cnt) < self.min_area:
                continue
            
            # Get the bounding box coordinates for each contour
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Dummy confidence value
            confidence = 1.0
            
            # Label all detections as generic "object"
            class_id = 0
            class_name = "object"
            
            detections.append([x, y, x + w, y + h, confidence, class_id, class_name])
        
        # Apply non-maximum suppression using only numeric data
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
        
        # Extract numeric values (first five elements) and convert to float
        boxes = np.array([d[:5] for d in detections], dtype=float)
        
        # Grab coordinates and confidence scores
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        # Compute the area of each bounding box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort indexes by confidence score (lowest to highest)
        idxs = np.argsort(scores)
        
        pick = []
        while len(idxs) > 0:
            # Grab the index with the highest score and add it to the pick list
            last = idxs[-1]
            pick.append(last)
            
            # Compute the coordinates for the intersection boxes
            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])
            
            # Compute width and height of the intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Compute the ratio of overlap (IoU)
            inter = w * h
            iou = inter / (areas[last] + areas[idxs[:-1]] - inter)
            
            # Delete all indexes that have IoU greater than the threshold
            idxs = np.delete(
                idxs,
                np.concatenate(([len(idxs)-1], np.where(iou > overlap_thresh)[0]))
            )
        
        # Return the detections corresponding to the picked indexes
        filtered_detections = [detections[i] for i in pick]
        return filtered_detections

# Example usage:
if __name__ == "__main__":
    image = cv2.imread("path_to_your_image.jpg")
    if image is None:
        print("Error: Image not found.")
        exit()

    detector = ObjectDetectorCV(min_area=500, nms_threshold=0.3)
    detections = detector.detect(image)
    
    for (x1, y1, x2, y2, conf, class_id, class_name) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
