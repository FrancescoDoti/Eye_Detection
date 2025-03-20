import cv2
import time
import numpy as np
from object_detector import ObjectDetectorCV  
from gaze_tracker import GazeTracker

class GazeDrivenObjectDetection:
    def __init__(self, window_name="Gaze-Driven Object Detection", 
                 confidence_threshold=1.0, gaze_influence=0.7):
        """
        Initialize the gaze-driven object detection system using classical CV methods.
        
        Args:
            window_name (str): Name of the display window.
            confidence_threshold (float): Dummy threshold for object detection confidence.
                                          (For classical methods, detections have a constant confidence of 1.0.)
            gaze_influence (float): Weight for how much gaze affects object prioritization (0-1).
        """
        self.window_name = window_name
        self.gaze_influence = gaze_influence
        
        # Initialize the classical object detector with a dummy confidence threshold.
        self.object_detector = ObjectDetectorCV(confidence_threshold)
        
        # Initialize the display window.
        cv2.namedWindow(window_name)
        
        # Initialize the gaze tracker.
        self.gaze_tracker = GazeTracker(window_name)
        
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
            
            # Combine the (dummy) detection confidence with the gaze attention.
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
        Process a single frame by detecting objects, generating a gaze heatmap, and visualizing results.
        
        Args:
            frame (numpy.ndarray): Input frame.
            
        Returns:
            numpy.ndarray: Frame with visualized detections and gaze information.
        """
        # Create a copy of the frame for visualization.
        visualization = frame.copy()
        
        # Detect objects using the classical detector.
        start_time = time.time()
        detections = self.object_detector.detect(frame)
        self.detection_times.append(time.time() - start_time)
        
        # Get the gaze heatmap from the gaze tracker.
        gaze_heatmap = self.gaze_tracker.generate_gaze_heatmap(frame.shape)
        
        # Prioritize detections based on the gaze heatmap.
        prioritized_detections = self.prioritize_detections(detections, gaze_heatmap)
        
        # Visualize the gaze heatmap.
        heatmap_visualization = cv2.applyColorMap((gaze_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        visualization = cv2.addWeighted(visualization, 0.7, heatmap_visualization, 0.3, 0)
        
        # Visualize detections with colors indicating priority (red = highest priority, green = lower).
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]
        
        for i, detection in enumerate(prioritized_detections):
            x1, y1, x2, y2, score, class_id, class_name = detection
            
            # Choose a color based on detection priority.
            color = colors[min(i, len(colors) - 1)]
            
            # Draw the bounding box.
            cv2.rectangle(visualization, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw the label.
            label = f"{class_name}: {score:.2f}"
            cv2.putText(visualization, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw the gaze position.
        gaze_x, gaze_y = self.gaze_tracker.get_gaze_position()
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
        
        print("Starting gaze-driven object detection...")
        print("Move your mouse to simulate gaze position.")
        print("Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            
            result = self.process_frame(frame)
            cv2.imshow(self.window_name, result)
            
            # Exit if 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics.
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        avg_prioritization_time = np.mean(self.prioritization_times) if self.prioritization_times else 0
        print(f"Average detection time: {avg_detection_time*1000:.2f} ms")
        print(f"Average prioritization time: {avg_prioritization_time*1000:.2f} ms")
        print(f"Total average processing time: {(avg_detection_time + avg_prioritization_time)*1000:.2f} ms")
