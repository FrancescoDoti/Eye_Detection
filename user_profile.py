class UserProfile:
    def __init__(self, user_id, name):
        """
        Initialize a user profile for personalized gaze-driven object detection
        
        Args:
            user_id (str): Unique identifier for the user
            name (str): User's name
        """
        self.user_id = user_id
        self.name = name
        
        # User-specific parameters
        self.gaze_sigma = 50  # Standard deviation for gaze heatmap
        self.gaze_influence = 0.7  # How much gaze affects prioritization
        
        # History of interactions
        self.object_interaction_history = {}  # {class_name: count}
        
        # User preferences (e.g., objects of interest)
        self.preferred_objects = []
    
    def update_interaction(self, object_class):
        """
        Update interaction history for an object class
        
        Args:
            object_class (str): Class name of the object
        """
        if object_class in self.object_interaction_history:
            self.object_interaction_history[object_class] += 1
        else:
            self.object_interaction_history[object_class] = 1
    
    def get_object_preference_score(self, object_class):
        """
        Get preference score for an object class based on interaction history
        
        Args:
            object_class (str): Class name of the object
            
        Returns:
            float: Preference score (0-1)
        """
        if not self.object_interaction_history:
            return 0.5  # Neutral if no history
        
        # Get count for this object class
        count = self.object_interaction_history.get(object_class, 0)
        
        # Get maximum count across all object classes
        max_count = max(self.object_interaction_history.values()) if self.object_interaction_history else 1
        
        # Normalize count to 0-1 range
        score = count / max_count if max_count > 0 else 0
        
        return score
    
    def add_preferred_object(self, object_class):
        """
        Add an object class to the list of preferred objects
        
        Args:
            object_class (str): Class name of the object
        """
        if object_class not in self.preferred_objects:
            self.preferred_objects.append(object_class)
    
    def remove_preferred_object(self, object_class):
        """
        Remove an object class from the list of preferred objects
        
        Args:
            object_class (str): Class name of the object
        """
        if object_class in self.preferred_objects:
            self.preferred_objects.remove(object_class)
    
    def is_preferred_object(self, object_class):
        """
        Check if an object class is in the list of preferred objects
        
        Args:
            object_class (str): Class name of the object
            
        Returns:
            bool: True if the object is preferred, False otherwise
        """
        return object_class in self.preferred_objects