"""
Bullet Impact Detection System
Detects and tracks bullet impacts on a static paper target using computer vision.
"""

import cv2
import numpy as np
from datetime import datetime
from collections import deque


class BulletImpactDetector:
    """
    Detects bullet impacts on a paper target from video feed using frame differencing
    and contour detection.
    """
    
    def __init__(self, video_source=0, min_hole_area=50, max_hole_area=2000):
        """
        Initialize the bullet impact detector.
        
        Args:
            video_source: Camera index or video file path
            min_hole_area: Minimum area (pixels) for a valid bullet hole
            max_hole_area: Maximum area (pixels) for a valid bullet hole
        """
        self.cap = cv2.VideoCapture(video_source)
        self.reference_frame = None
        self.baseline_captured = False
        self.impact_locations = []
        self.min_hole_area = min_hole_area
        self.max_hole_area = max_hole_area
        self.frame_buffer = deque(maxlen=5)  # Buffer for temporal filtering
        self.detection_cooldown = 0
        
        # Detection parameters
        self.blur_kernel = (5, 5)
        self.diff_threshold = 30
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def preprocess_frame(self, frame):
        """
        Preprocess frame for analysis.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed grayscale frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        return blurred
    
    def capture_baseline(self, frame):
        """
        Capture the baseline reference image of the clean target.
        
        Args:
            frame: Clean target frame (before any shots)
        """
        self.reference_frame = self.preprocess_frame(frame)
        self.baseline_captured = True
        print("Baseline captured. Ready to detect impacts.")
    
    def detect_new_impacts(self, current_frame):
        """
        Detect new bullet impacts by comparing current frame with reference.
        
        Args:
            current_frame: Current video frame
            
        Returns:
            List of detected impact locations [(x, y, area), ...]
        """
        if not self.baseline_captured:
            return []
        
        # Cooldown to prevent duplicate detections
        if self.detection_cooldown > 0:
            self.detection_cooldown -= 1
            return []
        
        processed = self.preprocess_frame(current_frame)
        
        # Compute frame difference
        frame_diff = cv2.absdiff(self.reference_frame, processed)
        
        # Apply threshold
        _, thresh = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        new_impacts = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_hole_area <= area <= self.max_hole_area:
                # Calculate circularity (bullet holes are roughly circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Filter by circularity (0.5-1.0 for reasonably circular shapes)
                if circularity > 0.3:
                    # Get center point
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if this is a new impact (not too close to existing ones)
                        if self._is_new_impact(cx, cy):
                            new_impacts.append({
                                'x': cx,
                                'y': cy,
                                'area': area,
                                'circularity': circularity,
                                'timestamp': datetime.now(),
                                'contour': contour
                            })
        
        return new_impacts
    
    def _is_new_impact(self, x, y, min_distance=30):
        """
        Check if detected impact is new (not near existing impacts).
        
        Args:
            x, y: Coordinates of potential new impact
            min_distance: Minimum distance to existing impacts
            
        Returns:
            True if this is a new impact, False otherwise
        """
        for impact in self.impact_locations:
            distance = np.sqrt((impact['x'] - x)**2 + (impact['y'] - y)**2)
            if distance < min_distance:
                return False
        return True
    
    def update_reference(self, frame):
        """
        Update reference frame to include new detected holes.
        This prevents re-detecting the same hole.
        
        Args:
            frame: Current frame with detected holes
        """
        self.reference_frame = self.preprocess_frame(frame)
    
    def draw_impacts(self, frame):
        """
        Draw detected impacts on the frame.
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw all detected impacts
        for i, impact in enumerate(self.impact_locations, 1):
            x, y = impact['x'], impact['y']
            
            # Draw circle around impact
            cv2.circle(annotated, (x, y), 10, (0, 0, 255), 2)
            
            # Draw center point
            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
            
            # Draw impact number
            cv2.putText(annotated, f"#{i}", (x + 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display statistics
        stats_text = f"Total Impacts: {len(self.impact_locations)}"
        cv2.putText(annotated, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def run(self):
        """
        Main detection loop.
        """
        dobaseline = True
        
        print("Bullet Impact Detector Started")
        print("Press 'b' to capture baseline (clean target)")
        print("Press 'r' to reset detections")
        print("Press 's' to save current frame")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            #automatic baseline capture
            if dobaseline:
                self.capture_baseline(frame)
                dobaseline = False
                
            if not ret:
                print("Failed to capture frame")
                break
            
            # Show baseline capture status
            if not self.baseline_captured:
                cv2.putText(frame, "Press 'b' to capture baseline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Bullet Impact Detector', frame)
            else:
                # Detect new impacts
                new_impacts = self.detect_new_impacts(frame)
                
                if new_impacts:
                    print(f"\n{len(new_impacts)} new impact(s) detected!")
                    for impact in new_impacts:
                        self.impact_locations.append(impact)
                        print(f"  Impact #{len(self.impact_locations)}: "
                        f"Position ({impact['x']}, {impact['y']}), "
                        f"Area: {impact['area']:.0f}px, "
                        f"Circularity: {impact['circularity']:.2f}")
                    
                    # Update reference to include new holes
                    self.update_reference(frame)
                    
                    # Set cooldown to prevent duplicate detections
                    self.detection_cooldown = 10
                
                # Draw annotations
                display_frame = self.draw_impacts(frame)
                cv2.imshow('Bullet Impact Detector', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('b'):
                self.capture_baseline(frame)
            elif key == ord('r'):
                self.impact_locations = []
                self.baseline_captured = False
                print("Detections reset. Capture new baseline.")
            elif key == ord('s'):
                filename = f"impact_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, self.draw_impacts(frame))
                print(f"Frame saved as {filename}")
            elif key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nFinal Statistics:")
        print(f"Total impacts detected: {len(self.impact_locations)}")
        
        if self.impact_locations:
            print("\nImpact Log:")
            for i, impact in enumerate(self.impact_locations, 1):
                print(f"  Shot #{i}: ({impact['x']}, {impact['y']}) at "
                      f"{impact['timestamp'].strftime('%H:%M:%S.%f')[:-3]}")


def main():
    """
    Main entry point for the application.
    """
    # Initialize detector
    # Use 0 for webcam, or provide path to video file
    detector = BulletImpactDetector(
        video_source="sample2.mp4",
        min_hole_area=50,
        max_hole_area=2000
    )
    
    # Run detection loop
    detector.run()


if __name__ == "__main__":
    main()
