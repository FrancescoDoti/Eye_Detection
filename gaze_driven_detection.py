import cv2
import time
import numpy as np
from object_detector import ObjectDetector
from gaze_tracker import GazeTracker


class GazeDrivenObjectDetection:
    def __init__(self, window_name="Gaze-Driven Object Detection", 
                 confidence_threshold=0.5, gaze_influence=0.7):
        """
        Initialize the gaze-driven object detection system
        
        Args:
            window_name (str): Name of the display window
            confidence_threshold (float): Threshold for object detection confidence
            gaze_influence (float): How much gaze affects object prioritization (0-1)
        """
        self.window_name = window_name
        self.gaze_influence = gaze_influence
        
        # Initialize the object detector
        self.object_detector = ObjectDetector(confidence_threshold)
        
        # Initialize the display window
        cv2.namedWindow(window_name)
        
        # Initialize the gaze tracker
        self.gaze_tracker = GazeTracker(window_name)
        
        # Performance metrics
        self.detection_times = []
        self.prioritization_times = []
    
    def prioritize_detections(self, detections, gaze_heatmap):
        """
        Prioritize detections based on gaze heatmap
        
        Args:
            detections (list): List of detections from object detector
            gaze_heatmap (numpy.ndarray): Gaze heatmap
            
        Returns:
            list: Prioritized detections
        """
        start_time = time.time()
        
        prioritized_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, score, class_id, class_name = detection
            
            # Calculate box center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get gaze attention at center of box
            center_x_int, center_y_int = int(center_x), int(center_y)
            if 0 <= center_y_int < gaze_heatmap.shape[0] and 0 <= center_x_int < gaze_heatmap.shape[1]:
                gaze_attention = gaze_heatmap[center_y_int, center_x_int]
            else:
                gaze_attention = 0
            
            # Combine detection confidence with gaze attention
            combined_score = (1 - self.gaze_influence) * score + self.gaze_influence * gaze_attention
            
            # Update detection with combined score
            prioritized_detection = [x1, y1, x2, y2, combined_score, class_id, class_name]
            prioritized_detections.append(prioritized_detection)
        
        # Sort detections by combined score (descending)
        prioritized_detections.sort(key=lambda x: x[4], reverse=True)
        
        self.prioritization_times.append(time.time() - start_time)
        
        return prioritized_detections
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame with visualizations
        """
        # Create a copy of the frame for visualization
        visualization = frame.copy()
        
        # Detect objects
        start_time = time.time()
        detections = self.object_detector.detect(frame)
        self.detection_times.append(time.time() - start_time)
        
        # Get gaze position and generate heatmap
        gaze_heatmap = self.gaze_tracker.generate_gaze_heatmap(frame.shape)
        
        # Prioritize detections based on gaze
        prioritized_detections = self.prioritize_detections(detections, gaze_heatmap)
        
        # Visualize gaze heatmap
        heatmap_visualization = cv2.applyColorMap((gaze_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        visualization = cv2.addWeighted(visualization, 0.7, heatmap_visualization, 0.3, 0)
        
        # Visualize detections with different colors based on priority
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]  # Red to green
        
        for i, detection in enumerate(prioritized_detections):
            x1, y1, x2, y2, score, class_id, class_name = detection
            
            # Get color based on priority (first has highest priority)
            color_idx = min(i, len(colors) - 1)
            color = colors[color_idx]
            
            # Draw bounding box
            cv2.rectangle(visualization, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(visualization, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw gaze position
        gaze_x, gaze_y = self.gaze_tracker.get_gaze_position()
        cv2.circle(visualization, (gaze_x, gaze_y), 10, (255, 0, 0), -1)
        
        return visualization
    
    def run(self, source=0):
        """
        Run the gaze-driven object detection system
        
        Args:
            source: Video source (0 for webcam, or path to video file)
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        print("Starting gaze-driven object detection...")
        print("Move your mouse to simulate gaze position.")
        print("Press 'q' to quit.")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream.")
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Display result
            cv2.imshow(self.window_name, result)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance statistics
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        avg_prioritization_time = np.mean(self.prioritization_times) if self.prioritization_times else 0
        
        print(f"Average detection time: {avg_detection_time*1000:.2f} ms")
        print(f"Average prioritization time: {avg_prioritization_time*1000:.2f} ms")
        print(f"Total average processing time: {(avg_detection_time + avg_prioritization_time)*1000:.2f} ms")