import time
import numpy as np
from gaze_driven_detection import GazeDrivenObjectDetectionCV  # Classical CV version

class ContextAwareObjectDetectionCV(GazeDrivenObjectDetectionCV):
    def __init__(self, window_name="Context-Aware Object Detection", 
                 min_area=500, gaze_influence=0.7):
        """
        Initialize the context-aware object detection system using classical CV methods
        
        Args:
            window_name (str): Name of the display window.
            min_area (int): Minimum area threshold for detecting objects via classical methods.
            gaze_influence (float): How much gaze affects object prioritization (0-1).
        """
        # For classical methods, detection confidence is often a dummy constant (e.g. 1.0)
        super().__init__(window_name, confidence_threshold=1.0, gaze_influence=gaze_influence)
        
        # Classical detector parameter
        self.min_area = min_area
        
        # Set the default context (e.g., "driving", "kitchen", "living_room")
        self.current_context = "general"
        
        # Define context-specific object importance scores
        self.context_importance = {
            "driving": {
                "person": 1.0,
                "car": 0.9,
                "traffic light": 0.9,
                "stop sign": 0.9,
                "truck": 0.8,
                "motorcycle": 0.8,
                "bicycle": 0.8,
                "bus": 0.7
            },
            "kitchen": {
                "person": 1.0,
                "cup": 0.8,
                "bowl": 0.8,
                "bottle": 0.7,
                "fork": 0.7,
                "knife": 0.7,
                "spoon": 0.7,
                "microwave": 0.6,
                "oven": 0.6,
                "refrigerator": 0.6,
                "sink": 0.6
            },
            "living_room": {
                "person": 1.0,
                "tv": 0.8,
                "laptop": 0.8,
                "remote": 0.7,
                "cell phone": 0.7,
                "couch": 0.6,
                "chair": 0.6,
                "book": 0.5
            },
            "general": {}  # Default context with no specific importance settings
        }
        
        # Current user profile (to incorporate user preferences)
        self.current_user = None
        
        # List to track prioritization processing times for evaluation
        self.prioritization_times = []
    
    def set_context(self, context):
        """
        Set the current context.
        
        Args:
            context (str): Context name (e.g., "driving", "kitchen", "living_room").
        """
        if context in self.context_importance:
            self.current_context = context
            print(f"Context set to: {context}")
        else:
            print(f"Unknown context: {context}. Using 'general' context.")
            self.current_context = "general"
    
    def set_user(self, user):
        """
        Set the current user.
        
        Args:
            user (UserProfile): User profile object.
        """
        self.current_user = user
        print(f"User set to: {user.name}")
    
    def get_context_importance(self, class_name):
        """
        Retrieve the context-specific importance for an object class.
        
        Args:
            class_name (str): Name of the object class.
            
        Returns:
            float: Importance score between 0 and 1 (default is 0.5 if not specified).
        """
        return self.context_importance.get(self.current_context, {}).get(class_name, 0.5)
    
    def prioritize_detections(self, detections, gaze_heatmap):
        """
        Prioritize detections based on the gaze heatmap, context importance, and user preferences.
        
        Args:
            detections (list): List of detections from the classical object detector.
                               Each detection is [x1, y1, x2, y2, confidence, class_id, class_name].
            gaze_heatmap (numpy.ndarray): Gaze heatmap as a 2D array of intensity values.
            
        Returns:
            list: Detections sorted by the combined score (descending order).
        """
        start_time = time.time()
        prioritized_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, score, class_id, class_name = detection
            
            # Calculate the center of the detection box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get gaze attention at the center (safeguarding for bounds)
            center_x_int, center_y_int = int(center_x), int(center_y)
            if (0 <= center_y_int < gaze_heatmap.shape[0] and 
                0 <= center_x_int < gaze_heatmap.shape[1]):
                gaze_attention = gaze_heatmap[center_y_int, center_x_int]
            else:
                gaze_attention = 0
            
            # Retrieve context importance for the detected object's class
            context_importance = self.get_context_importance(class_name)
            
            # Retrieve user preference score (default to 0.5 if no user profile is set)
            user_preference = 0.5
            if self.current_user:
                user_preference = self.current_user.get_object_preference_score(class_name)
                if self.current_user.is_preferred_object(class_name):
                    user_preference = max(user_preference, 0.8)
            
            # Combine the factors:
            #  - 30% weight for the detection confidence (often a constant value in classical detection)
            #  - 40% weight for gaze attention (proximity to user gaze)
            #  - 20% weight for context importance (importance of object in the current scene)
            #  - 10% weight for user preference (user-specific boosting)
            combined_score = (
                0.3 * score +
                0.4 * gaze_attention +
                0.2 * context_importance +
                0.1 * user_preference
            )
            
            # Update the detection with the combined score
            prioritized_detection = [x1, y1, x2, y2, combined_score, class_id, class_name]
            prioritized_detections.append(prioritized_detection)
        
        # Sort detections by the combined score in descending order
        prioritized_detections.sort(key=lambda d: d[4], reverse=True)
        
        # Record the processing time for prioritization
        self.prioritization_times.append(time.time() - start_time)
        
        return prioritized_detections
