import cv2
import numpy as np


class GazeTracker:
    def __init__(self, window_name, use_mouse=True):
        """
        Initialize the gaze tracker
        
        Args:
            window_name (str): Name of the window to capture mouse events from
            use_mouse (bool): Whether to use mouse position as a proxy for gaze
        """
        self.window_name = window_name
        self.use_mouse = use_mouse
        self.gaze_position = (0, 0)
        
        # Set up mouse callback if using mouse as proxy for gaze
        if use_mouse:
            cv2.setMouseCallback(window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback function for mouse events"""
        self.gaze_position = (x, y)
    
    def get_gaze_position(self):
        """
        Get current gaze position
        
        Returns:
            tuple: (x, y) coordinates of gaze position
        """
        return self.gaze_position
    
    def get_eye_position(self):
        return self.get_gaze_position()
    
    def generate_gaze_heatmap(self, image_shape, sigma=50):
        """
        Generate a gaze heatmap based on current gaze position
        
        Args:
            image_shape (tuple): Shape of the image (height, width)
            sigma (int): Standard deviation for Gaussian kernel
            
        Returns:
            numpy.ndarray: Gaze heatmap
        """
        height, width = image_shape[:2]
        x, y = self.gaze_position
        
        # Create meshgrid
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # Create Gaussian heatmap
        heatmap = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        
        # Normalize heatmap
        heatmap = heatmap / np.max(heatmap)
        
        return heatmap

    # For integration with real eye trackers in the future
    def connect_to_eye_tracker(self, tracker_type="tobii"):
        """
        Connect to an actual eye tracker device
        
        Args:
            tracker_type (str): Type of eye tracker ('tobii' or 'pupil_labs')
            
        Returns:
            bool: Success status
        """
        # This would be implemented when actual hardware is available
        # For now, return False to indicate we're using the simulation
        return False