import os
import cv2
import numpy as np
import json

class ObjectDetectorCV:
    def __init__(self, min_area=500):
        """
        Initialize a simple classical object detector using contours.
        
        Args:
            min_area (int): Minimum area (in pixels) for a contour to be considered a detection.
        """
        self.min_area = min_area

    def detect(self, image):
        """
        Detect objects using edge detection and contour finding.
        
        Args:
            image (numpy.ndarray): Input image in BGR format.
            
        Returns:
            list: List of detections, each formatted as 
                  [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            confidence = 1.0  # Dummy confidence value
            class_id = 0
            class_name = "object"
            detections.append([x, y, x + w, y + h, confidence, class_id, class_name])
        return detections

class GazeDrivenObjectDetectionCV:
    def __init__(self, min_area=500, gaze_influence=0.7):
        """
        A gaze-driven detection system using classical techniques.
        
        Args:
            min_area (int): Minimum area for detections.
            gaze_influence (float): Factor to adjust detection scores based on gaze proximity.
        """
        self.detector = ObjectDetectorCV(min_area=min_area)
        self.gaze_influence = gaze_influence
        self.gaze_position = None

    def process_frame(self, image):
        """
        Process a frame by detecting objects and reweighting their confidence based on gaze.
        
        Args:
            image (numpy.ndarray): Input image in BGR format.
        
        Returns:
            numpy.ndarray: The input image with detection bounding boxes and labels drawn.
        """
        detections = self.detector.detect(image)
        # If a gaze position is provided, adjust the confidence scores of detections
        if self.gaze_position is not None:
            gaze_x, gaze_y = self.gaze_position
            adjusted_detections = []
            # Use the image diagonal as a normalization factor for distance
            diag = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
            for d in detections:
                x1, y1, x2, y2, conf, class_id, class_name = d
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance = np.sqrt((center_x - gaze_x)**2 + (center_y - gaze_y)**2)
                # Compute a weight that decreases with distance; detections near the gaze get a boost
                weight = 1 - self.gaze_influence * (distance / diag)
                new_conf = conf * weight
                adjusted_detections.append([x1, y1, x2, y2, new_conf, class_id, class_name])
            # Optionally, sort detections by the new confidence score
            detections = sorted(adjusted_detections, key=lambda d: d[4], reverse=True)
        
        # Draw the detections on the image
        for d in detections:
            x1, y1, x2, y2, conf, class_id, class_name = d
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

def test_with_dataset(image_folder, gaze_data_file=None):
    """
    Test the classical gaze-driven detection system with a dataset of images and optional gaze data.
    
    Args:
        image_folder (str): Path to folder containing images.
        gaze_data_file (str): Path to file containing gaze data (optional). Expected format is JSON mapping 
                              image filenames to [gaze_x, gaze_y] coordinates.
    """
    system = GazeDrivenObjectDetectionCV(min_area=500, gaze_influence=0.7)
    
    # Load image filenames from the dataset folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    # Load gaze data if available
    gaze_data = {}
    if gaze_data_file and os.path.exists(gaze_data_file):
        with open(gaze_data_file, 'r') as f:
            gaze_data = json.load(f)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # If gaze data is available for the image, update the gaze position
        if image_file in gaze_data:
            gaze_x, gaze_y = gaze_data[image_file]
            system.gaze_position = (gaze_x, gaze_y)
        else:
            system.gaze_position = None
        
        # Process the image frame
        result = system.process_frame(image)
        
        # Display the result
        cv2.imshow("Dataset Testing", result)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    # Replace with your actual image folder and gaze data file paths
    test_with_dataset("path_to_your_images", "path_to_your_gaze_data.json")
