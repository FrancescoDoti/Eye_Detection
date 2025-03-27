import cv2
import numpy as np

class ObjectDetectorCV:
    def __init__(self, 
                 min_area=500, 
                 nms_threshold=0.3, 
                 face_scaleFactor=1.1, 
                 face_minNeighbors=5,
                 dark_threshold=50,
                 light_min_area=None,
                 dark_min_area=None):
        """
        Initialize the detector with two pipelines:
          - Light object detection (for objects with higher brightness)
          - Dark object detection (for objects with lower brightness)
        
        Args:
            min_area (int): Default minimum area for detections (used if light_min_area or dark_min_area are not provided).\n
            nms_threshold (float): Overlap threshold for non-maximum suppression.
            face_scaleFactor (float): Scale factor for Haar cascade face detection.
            face_minNeighbors (int): Min neighbors for Haar cascade face detection.
            dark_threshold (int): Intensity threshold; below this an object is considered dark.
            light_min_area (int): Minimum area for light object contours (overrides min_area if provided).\n
            dark_min_area (int): Minimum area for dark object contours (overrides min_area if provided).
        """
        # Use min_area as default if specific thresholds are not provided
        self.light_min_area = light_min_area if light_min_area is not None else min_area
        self.dark_min_area = dark_min_area if dark_min_area is not None else min_area
        
        self.nms_threshold = nms_threshold
        self.dark_threshold = dark_threshold
        
        # Load Haar cascade for face detection (used in dark object detection pipeline)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face_scaleFactor = face_scaleFactor
        self.face_minNeighbors = face_minNeighbors

    def detect_light(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        img_h, img_w = image.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.light_min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            if w > 0.5 * img_w or h > 0.5 * img_h:
                continue
            
            # Check average intensity inside the bounding box
            object_region = gray[y:y+h, x:x+w]
            avg_intensity = np.mean(object_region)
            if avg_intensity < self.dark_threshold:
                # Skip this detection; will be handled in dark pipeline.
                continue
            
            confidence = min(area / 1000.0, 1.0)
            detections.append([x, y, x + w, y + h, confidence, 0, "object"])
        return detections

    def detect_dark(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.face_scaleFactor,
            minNeighbors=self.face_minNeighbors,
            minSize=(30, 30)
        )
        face_mask = np.ones(gray.shape, dtype="uint8") * 255
        for (fx, fy, fw, fh) in faces:
            pad = 10
            hair_extra = int(0.5 * fh)
            cv2.rectangle(face_mask, 
                          (max(fx - pad, 0), max(fy - hair_extra - pad, 0)), 
                          (min(fx + fw + pad, gray.shape[1]), min(fy + fh + pad, gray.shape[0])), 
                          0, thickness=-1)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_and(thresh, thresh, mask=face_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(opened.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.dark_min_area:
                continue
            
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (float(area) / hull_area) < 0.5:
                continue
            
            confidence = 1.0
            detections.append([x, y, x + w, y + h, confidence, 0, "dark_object"])
        return detections

    def detect(self, image):
        detections_light = self.detect_light(image)
        detections_dark = self.detect_dark(image)
        all_detections = detections_light + detections_dark
        return self.non_max_suppression(all_detections, self.nms_threshold)
    
    def non_max_suppression(self, detections, overlap_thresh=0.3):
        if len(detections) == 0:
            return []
        
        boxes = np.array([d[:5] for d in detections], dtype=float)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        idxs = np.argsort(scores)
        pick = []
        
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[last] + areas[idxs[:-1]] - inter)
            idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(iou > overlap_thresh)[0]))) 
            
        return [detections[i] for i in pick]
