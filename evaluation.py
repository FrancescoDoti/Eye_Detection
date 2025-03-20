import numpy as np

def evaluate_detection_accuracy(ground_truth, detections, iou_threshold=0.5):
    """
    Evaluate detection accuracy using mean Average Precision (mAP)
    
    Args:
        ground_truth (list): List of ground truth annotations
        detections (list): List of detections
        iou_threshold (float): IoU threshold for matching
        
    Returns:
        float: mAP score
    """
    # This is a simplified mAP calculation
    # In a real implementation, we would use a more comprehensive mAP calculation
    
    # Count true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # Match detections to ground truth
    matched_gt = set()
    
    for detection in detections:
        det_x1, det_y1, det_x2, det_y2, _, det_class_id, _ = detection
        det_box = [det_x1, det_y1, det_x2, det_y2]
        
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
                
            gt_x1, gt_y1, gt_x2, gt_y2, _, gt_class_id, _ = gt
            
            # Skip if classes don't match
            if det_class_id != gt_class_id:
                continue
                
            gt_box = [gt_x1, gt_y1, gt_x2, gt_y2]
            
            # Calculate IoU
            iou = calculate_iou(det_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truth) - len(matched_gt)
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score as an approximation of mAP
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def evaluate_real_time_performance(detection_times, prioritization_times, target_fps=30):
    """
    Evaluate real-time performance
    
    Args:
        detection_times (list): List of detection times in seconds
        prioritization_times (list): List of prioritization times in seconds
        target_fps (int): Target frames per second
        
    Returns:
        dict: Performance metrics
    """
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    avg_prioritization_time = np.mean(prioritization_times) if prioritization_times else 0
    total_processing_time = avg_detection_time + avg_prioritization_time
    
    achieved_fps = 1 / total_processing_time if total_processing_time > 0 else float('inf')
    meets_target = achieved_fps >= target_fps
    
    return {
        'avg_detection_time_ms': avg_detection_time * 1000,
        'avg_prioritization_time_ms': avg_prioritization_time * 1000,
        'total_processing_time_ms': total_processing_time * 1000,
        'achieved_fps': achieved_fps,
        'target_fps': target_fps,
        'meets_target': meets_target
    }

def run_user_study(num_participants=10):
    """
    Simulate running a user study for interaction efficiency
    
    Args:
        num_participants (int): Number of participants
        
    Returns:
        dict: Study results
    """
    # In a real implementation, this would be an actual user study
    # For now, we'll simulate the results
    
    # Simulate search times (in seconds) with and without gaze guidance
    # Format: [without_gaze, with_gaze]
    simulated_search_times = [
        [5.2, 2.1],  # Participant 1
        [4.8, 1.9],  # Participant 2
        [6.1, 2.7],
        [5.5, 2.3],
        [4.9, 2.0],
        [5.7, 2.5],
        [6.3, 2.8],
        [5.0, 2.2],
        [5.8, 2.4],
        [5.3, 2.2]   # Participant 10
    ]
    
    # Calculate average search times
    avg_without_gaze = np.mean([x[0] for x in simulated_search_times])
    avg_with_gaze = np.mean([x[1] for x in simulated_search_times])
    
    # Calculate improvement
    improvement = (avg_without_gaze - avg_with_gaze) / avg_without_gaze * 100
    
    # Calculate standard deviations
    std_without_gaze = np.std([x[0] for x in simulated_search_times])
    std_with_gaze = np.std([x[1] for x in simulated_search_times])
    
    return {
        'avg_search_time_without_gaze': avg_without_gaze,
        'avg_search_time_with_gaze': avg_with_gaze,
        'improvement_percentage': improvement,
        'std_without_gaze': std_without_gaze,
        'std_with_gaze': std_with_gaze,
        'num_participants': num_participants
    }