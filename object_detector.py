import cv2
import numpy as np

class ObjectDetectorCV:
    def __init__(self, min_area=500):
        """
        Initialize the object detector without deep learning.
        
        Args:
            min_area (int): Minimum area in pixels to consider a contour as a valid detection.
        """
        self.min_area = min_area

    def detect(self, image):
        """
        Detect objects in the given image using classical image processing techniques.
        
        Args:
            image (numpy.ndarray): Input image in BGR format (as read by OpenCV)
            
        Returns:
            list: List of detections, each containing [x1, y1, x2, y2, confidence, class_id, class_name]
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
            
            # Here we assign a dummy confidence value
            confidence = 1.0
            
            # Since we don't have a classifier, we label all detections as generic "object"
            class_id = 0
            class_name = "object"
            
            detections.append([x, y, x + w, y + h, confidence, class_id, class_name])
        
        return detections

# Example usage:
if __name__ == "__main__":
    # Read an image from file
    image = cv2.imread("path_to_your_image.jpg")
    
    # Initialize our classical object detector
    detector = ObjectDetectorCV(min_area=500)
    
    # Perform detection
    detections = detector.detect(image)
    
    # Draw bounding boxes on the image
    for (x1, y1, x2, y2, conf, class_id, class_name) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
