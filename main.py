from object_detector import ObjectDetector
from gaze_tracker import GazeTracker
from gaze_driven_detection import GazeDrivenObjectDetection
from user_profile import UserProfile
from context_aware_detection import ContextAwareObjectDetection
from evaluation import evaluate_real_time_performance, run_user_study

def main_enhanced():
    """Main function to run the enhanced context-aware object detection system"""
    print("Context-Aware Gaze-Driven Object Detection for Enhanced Accessibility")
    print("====================================================================")
    
    # Create the system
    system = ContextAwareObjectDetection(
        confidence_threshold=0.5,
        gaze_influence=0.7
    )
    
    # Create a user profile
    user = UserProfile(user_id="user1", name="John Doe")
    user.add_preferred_object("book")
    user.add_preferred_object("laptop")
    
    # Set user and context
    system.set_user(user)
    system.set_context("living_room")
    
    # Set video source
    video_source = 0  # Webcam
    
    # Run the system
    system.run(video_source)
    
    # Run evaluation
    performance = evaluate_real_time_performance(
        system.detection_times, 
        system.prioritization_times
    )
    
    print("\nSystem Performance:")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    
    # Simulate user study
    study_results = run_user_study()
    
    print("\nUser Study Results:")
    for key, value in study_results.items():
        print(f"  {key}: {value}")
    
    print("\nSystem terminated.")


def main():
    """Main function to run the gaze-driven object detection system"""
    print("Gaze-Driven Object Detection for Enhanced Accessibility")
    print("======================================================")
    
    # Create the system
    system = GazeDrivenObjectDetection(
        confidence_threshold=0.5,  # Object detection confidence threshold
        gaze_influence=0.7         # How much gaze affects prioritization
    )
    
    # Set video source
    # 0 for webcam, or provide path to a video file
    video_source = 0
    
    # Run the system
    system.run(video_source)
    
    print("System terminated.")


if __name__ == "__main__":
    # Uncomment one of the following lines to run the desired version
    main()  # Basic version
    # main_enhanced()  # Enhanced version with context awareness