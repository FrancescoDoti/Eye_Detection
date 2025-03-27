import cv2
import time
import numpy as np
from object_detector import ObjectDetectorCV  # Ensure ObjectDetectorCV also accepts min_area
from gaze_tracker import GazeTracker

class GazeDrivenObjectDetectionCV:
    def __init__(self, window_name="Gaze-Driven Object Detection", 
                 confidence_threshold=1.0, gaze_influence=0.7, min_area=500):
        """
        Initialize the gaze-driven object detection system using classical CV methods.
        
        Args:
            window_name (str): Name of the display window.
            confidence_threshold (float): Dummy threshold for object detection confidence.
                                          (For classical methods, detections have a constant confidence of 1.0.)
            gaze_influence (float): Weight for how much gaze affects object prioritization (0-1).
            min_area (int): Minimum area threshold for detecting eyes.
        """
        self.window_name = window_name
        self.gaze_influence = gaze_influence
        self.min_area = min_area
        
        # Initialize the classical object (eye) detector with the provided min_area.
        self.object_detector = ObjectDetectorCV(min_area=self.min_area)
        
        # Create the display window.
        cv2.namedWindow(window_name)
        
        # Initialize the gaze tracker (using mouse as proxy).
        self.gaze_tracker = GazeTracker(window_name, use_mouse=True)
        
        # Performance metrics.
        self.detection_times = []
        self.prioritization_times = []
    
    def prioritize_detections(self, detections, gaze_heatmap):
        """
        Prioritize detections based on the gaze heatmap.
        
        Args:
            detections (list): List of detections from the object detector.
                               Each detection is formatted as 
                               [x1, y1, x2, y2, confidence, class_id, class_name].
            gaze_heatmap (numpy.ndarray): Gaze heatmap.
            
        Returns:
            list: Detections with updated combined scores, sorted in descending order.
        """
        start_time = time.time()
        prioritized_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, score, class_id, class_name = detection
            
            # Calculate the center of the detection box.
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get gaze attention at the center (with bounds checking).
            center_x_int, center_y_int = int(center_x), int(center_y)
            if 0 <= center_y_int < gaze_heatmap.shape[0] and 0 <= center_x_int < gaze_heatmap.shape[1]:
                gaze_attention = gaze_heatmap[center_y_int, center_x_int]
            else:
                gaze_attention = 0
            
            # Combine the dummy detection confidence with the gaze attention.
            combined_score = (1 - self.gaze_influence) * score + self.gaze_influence * gaze_attention
            
            # Update the detection with the combined score.
            prioritized_detection = [x1, y1, x2, y2, combined_score, class_id, class_name]
            prioritized_detections.append(prioritized_detection)
        
        # Sort detections by combined score (highest first).
        prioritized_detections.sort(key=lambda d: d[4], reverse=True)
        self.prioritization_times.append(time.time() - start_time)
        
        return prioritized_detections
    
    def process_frame(self, frame):
        """
        Process a single frame by detecting objects, generating a gaze heatmap, 
        and visualizing results based on whether the gaze (mouse) is inside a detection.
        
        Args:
            frame (numpy.ndarray): Input frame.
            
        Returns:
            numpy.ndarray: Frame with visualized detections and gaze information.
        """
        # Copy frame for visualization.
        visualization = frame.copy()
        
        # Detect objects using the classical detector.
        start_time = time.time()
        detections = self.object_detector.detect(frame)
        self.detection_times.append(time.time() - start_time)
        
        # Generate gaze heatmap.
        gaze_heatmap = self.gaze_tracker.generate_gaze_heatmap(frame.shape)
        
        # Prioritize detections based on the gaze heatmap.
        prioritized_detections = self.prioritize_detections(detections, gaze_heatmap)
        
        # Visualize the gaze heatmap.
        heatmap_visualization = cv2.applyColorMap((gaze_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        visualization = cv2.addWeighted(visualization, 0.7, heatmap_visualization, 0.3, 0)
        
        # Retrieve the current gaze (mouse) position.
        gaze_x, gaze_y = self.gaze_tracker.get_gaze_position()
        
        # Only draw detection boxes if the gaze pointer lies within them.
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]
        for i, detection in enumerate(prioritized_detections):
            x1, y1, x2, y2, score, class_id, class_name = detection
            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                color = colors[min(i, len(colors) - 1)]
                cv2.rectangle(visualization, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name}: {score:.2f}"
                cv2.putText(visualization, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw the current gaze position as a circle.
        cv2.circle(visualization, (gaze_x, gaze_y), 10, (255, 0, 0), -1)
        
        return visualization
    
    def run(self, source=0):
        """
        Run the gaze-driven object detection system.
        
        Args:
            source: Video source (0 for webcam, or path to a video file).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        print("Starting eye tracking...")
        print("Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            
            result = self.process_frame(frame)
            cv2.imshow(self.window_name, result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        print(f"Average detection time: {avg_detection_time*1000:.2f} ms")