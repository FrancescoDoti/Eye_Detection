import time
from gaze_driven_detection import GazeDrivenObjectDetection

class ContextAwareObjectDetection(GazeDrivenObjectDetection):
    def __init__(self, window_name="Context-Aware Object Detection", 
                 confidence_threshold=0.5, gaze_influence=0.7):
        """
        Initialize the context-aware object detection system
        
        Args:
            window_name (str): Name of the display window
            confidence_threshold (float): Threshold for object detection confidence
            gaze_influence (float): How much gaze affects object prioritization (0-1)
        """
        super().__init__(window_name, confidence_threshold, gaze_influence)
        
        # Current context (e.g., "driving", "kitchen", "living_room")
        self.current_context = "general"
        
        # Context-specific object importance
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
            "general": {}  # Default context with no specific importance
        }
        
        # Current user profile
        self.current_user = None
    
    def set_context(self, context):
        """
        Set the current context
        
        Args:
            context (str): Context name
        """
        if context in self.context_importance:
            self.current_context = context
            print(f"Context set to: {context}")
        else:
            print(f"Unknown context: {context}. Using 'general' context.")
            self.current_context = "general"
    
    def set_user(self, user):
        """
        Set the current user
        
        Args:
            user (UserProfile): User profile
        """
        self.current_user = user
        print(f"User set to: {user.name}")
    
    def get_context_importance(self, class_name):
        """
        Get context-specific importance for an object class
        
        Args:
            class_name (str): Class name of the object
            
        Returns:
            float: Importance score (0-1)
        """
        if self.current_context in self.context_importance:
            return self.context_importance[self.current_context].get(class_name, 0.5)
        return 0.5  # Neutral if context not found
    
    def prioritize_detections(self, detections, gaze_heatmap):
        """
        Prioritize detections based on gaze heatmap, context, and user profile
        
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
            
            # Get context importance
            context_importance = self.get_context_importance(class_name)
            
            # Get user preference (if available)
            user_preference = 0.5  # Neutral by default
            if self.current_user:
                user_preference = self.current_user.get_object_preference_score(class_name)
                
                # Boost score for preferred objects
                if self.current_user.is_preferred_object(class_name):
                    user_preference = max(user_preference, 0.8)
            
            # Combine scores
            # - Detection confidence: how confident the model is about the detection
            # - Gaze attention: how close the object is to where the user is looking
            # - Context importance: how important the object is in the current context
            # - User preference: how much the user has interacted with this type of object
            
            combined_score = (
                0.3 * score +                # Detection confidence
                0.4 * gaze_attention +       # Gaze attention
                0.2 * context_importance +   # Context importance
                0.1 * user_preference        # User preference
            )
            
            # Update detection with combined score
            prioritized_detection = [x1, y1, x2, y2, combined_score, class_id, class_name]
            prioritized_detections.append(prioritized_detection)
        
        # Sort detections by combined score (descending)
        prioritized_detections.sort(key=lambda x: x[4], reverse=True)
        
        self.prioritization_times.append(time.time() - start_time)
        
        return prioritized_detections