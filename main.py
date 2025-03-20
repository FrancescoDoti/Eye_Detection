from object_detector import ObjectDetectorCV
from gaze_tracker import GazeTracker  
from gaze_driven_detection import GazeDrivenObjectDetectionCV
from user_profile import UserProfile
from context_aware_detection import ContextAwareObjectDetectionCV
from evaluation import evaluate_real_time_performance, run_user_study

def main_enhanced():
    """Main function to run the enhanced context-aware object detection system using classical CV"""
    print("Context-Aware Gaze-Driven Object Detection for Enhanced Accessibility (Classical CV)")
    print("====================================================================")
    
    # Create the system using the classical implementation
    system = ContextAwareObjectDetectionCV(
        min_area=500,         # Replace deep learning detection with contour-based detection
        gaze_influence=0.7
    )
    
    # Create a user profile
    user = UserProfile(user_id="user1", name="John Doe")
    user.add_preferred_object("book")
    user.add_preferred_object("laptop")
    
    # Set user profile and context information
    system.set_user(user)
    system.set_context("living_room")
    
    # Set video source (0 for webcam, or path to a video file)
    video_source = 0  
    
    # Run the system (this method should open the video stream, process frames,
    # and display results using classical CV methods)
    system.run(video_source)
    
    # Run evaluation if the system collects performance metrics
    performance = evaluate_real_time_performance(
        system.detection_times, 
        system.prioritization_times
    )
    
    print("\nSystem Performance:")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    
    # Simulate a user study (this function may simulate responses or gather user feedback)
    study_results = run_user_study()
    
    print("\nUser Study Results:")
    for key, value in study_results.items():
        print(f"  {key}: {value}")
    
    print("\nSystem terminated.")


def main():
    """Main function to run the gaze-driven object detection system using classical CV"""
    print("Gaze-Driven Object Detection for Enhanced Accessibility (Classical CV)")
    print("======================================================")
    
    # Create the system using the classical implementation
    system = GazeDrivenObjectDetectionCV(
        min_area=500,         # Use contour-based detection instead of deep learning
        gaze_influence=0.7
    )
    
    # Set video source (0 for webcam, or provide a video file path)
    video_source = 0
    
    # Run the system
    system.run(video_source)
    
    print("System terminated.")


if __name__ == "__main__":
    # Uncomment one of the following lines to run the desired version
    main()            # Basic version using classical gaze-driven detection
    # main_enhanced()  # Enhanced version with added context awareness
