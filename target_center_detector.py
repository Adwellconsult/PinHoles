"""
Target Center Detection System
Identifies the center of concentric circle paper targets in video using computer vision.
"""

import cv2
import numpy as np
from datetime import datetime
import math


class TargetCenterDetector:
    """
    Detects the center of concentric circle targets in video frames.
    """
    
    def __init__(self, video_source="sample2.mp4", target_diameter_mm=200):
        """
        Initialize the target center detector.
        
        Args:
            video_source: Path to video file or camera index
            target_diameter_mm: Expected target diameter in millimeters
        """
        self.cap = cv2.VideoCapture(video_source)
        self.target_center = None
        self.confidence = 0.0
        self.detected_circles = []
        
        # Target specifications
        self.target_diameter_mm = target_diameter_mm
        self.pixels_per_mm = 1.0  # Will be calibrated
        self.target_diameter_pixels = 200  # Initial estimate
        
        # Detection parameters (will be auto-adjusted based on target size)
        self.min_radius = 20
        self.max_radius = 300
        self.dp = 1
        self.min_dist = 50
        self.param1 = 50
        self.param2 = 30
        
        self.update_detection_parameters()
        
    def preprocess_frame(self, frame):
        """
        Preprocess frame for circle detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed grayscale frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def update_detection_parameters(self):
        """
        Update detection parameters based on target diameter.
        """
        # Estimate target radius in pixels
        target_radius = self.target_diameter_pixels // 2
        
        # Set detection range based on target size
        self.min_radius = max(10, target_radius // 10)  # Smallest ring
        self.max_radius = min(500, int(target_radius * 1.2))  # Slightly larger than target
        
        # Minimum distance between circles (prevent detecting same circle multiple times)
        self.min_dist = max(20, target_radius // 8)
        
        print(f"Updated detection parameters for {self.target_diameter_mm}mm target:")
        print(f"  Target radius estimate: {target_radius} pixels")
        print(f"  Detection range: {self.min_radius}-{self.max_radius} pixels")
        print(f"  Minimum distance: {self.min_dist} pixels")
    
    def set_target_diameter(self, diameter_mm):
        """
        Set new target diameter and update detection parameters.
        
        Args:
            diameter_mm: Target diameter in millimeters
        """
        self.target_diameter_mm = diameter_mm
        # Estimate pixels based on current detection or default
        if self.target_center and self.detected_circles:
            # Use largest detected circle as reference
            largest_circle = max(self.detected_circles, key=lambda c: c['radius'])
            self.target_diameter_pixels = largest_circle['radius'] * 2
        else:
            # Default estimate (adjust based on video resolution)
            self.target_diameter_pixels = 200
        
        self.update_detection_parameters()
    
    def calibrate_pixel_scale(self, selected_circle_radius):
        """
        Calibrate pixel-to-mm scale based on user-selected circle.
        
        Args:
            selected_circle_radius: Radius of user-selected circle in pixels
        """
        self.target_diameter_pixels = selected_circle_radius * 2
        self.pixels_per_mm = self.target_diameter_pixels / self.target_diameter_mm
        self.update_detection_parameters()
        print(f"Calibrated: {self.pixels_per_mm:.2f} pixels per mm")
    
    def detect_concentric_circles(self, gray_frame):
        """
        Detect concentric circles in the frame.
        
        Args:
            gray_frame: Preprocessed grayscale frame
            
        Returns:
            List of detected circles and their centers
        """
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray_frame,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_circles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                detected_circles.append({
                    'center': (x, y),
                    'radius': r,
                    'area': math.pi * r * r
                })
        
        return detected_circles
    
    def find_concentric_center(self, circles):
        """
        Find the center of concentric circles by analyzing circle relationships.
        
        Args:
            circles: List of detected circles
            
        Returns:
            Estimated target center and confidence score
        """
        if len(circles) < 2:
            return None, 0.0
        
        # Sort circles by radius
        circles = sorted(circles, key=lambda c: c['radius'])
        
        # Group circles that are roughly concentric
        concentric_groups = []
        center_tolerance = 30  # pixels
        
        for circle in circles:
            added_to_group = False
            
            for group in concentric_groups:
                # Check if this circle is concentric with the group
                group_center = self.calculate_group_center(group)
                if group_center is not None:
                    dist = np.sqrt((circle['center'][0] - group_center[0])**2 + 
                                 (circle['center'][1] - group_center[1])**2)
                    
                    if dist < center_tolerance:
                        group.append(circle)
                        added_to_group = True
                        break
            
            if not added_to_group:
                concentric_groups.append([circle])
        
        # Find the group with the most circles (most likely the target)
        best_group = max(concentric_groups, key=len) if concentric_groups else []
        
        if len(best_group) < 2:
            return None, 0.0
        
        # Calculate the center of the best concentric group
        center = self.calculate_group_center(best_group)
        confidence = min(len(best_group) / 5.0, 1.0)  # More circles = higher confidence
        
        return center, confidence
    
    def calculate_group_center(self, circles):
        """
        Calculate the weighted center of a group of circles.
        
        Args:
            circles: List of circles in the group
            
        Returns:
            Weighted center point
        """
        if not circles:
            return None
        
        # Weight by inverse of radius (smaller circles are often more accurate)
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for circle in circles:
            # Use inverse radius as weight (smaller circles get higher weight)
            weight = 1.0 / (circle['radius'] + 1)
            weighted_x += circle['center'][0] * weight
            weighted_y += circle['center'][1] * weight
            total_weight += weight
        
        if total_weight > 0:
            return (int(weighted_x / total_weight), int(weighted_y / total_weight))
        
        return None
    
    def detect_target_center(self, frame):
        """
        Main method to detect target center in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Target center coordinates and confidence
        """
        # Preprocess frame
        gray = self.preprocess_frame(frame)
        
        # Detect circles
        circles = self.detect_concentric_circles(gray)
        
        # Find concentric center
        center, confidence = self.find_concentric_center(circles)
        
        # Store results
        self.detected_circles = circles
        if center is not None and confidence > 0.3:  # Only update if confident
            self.target_center = center
            self.confidence = confidence
        
        return center, confidence
    
    def draw_detections(self, frame):
        """
        Draw detected circles and target center on frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame
        """
        result = frame.copy()
        
        # Draw detected circles
        for circle in self.detected_circles:
            center = circle['center']
            radius = circle['radius']
            
            # Draw circle outline
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result, center, 3, (0, 255, 0), -1)
        
        # Draw target center if detected
        if self.target_center is not None:
            # Draw crosshairs
            cv2.drawMarker(result, self.target_center, (0, 0, 255), 
                          cv2.MARKER_CROSS, 30, 3)
            
            # Draw confidence circle
            confidence_radius = int(20 * self.confidence)
            cv2.circle(result, self.target_center, confidence_radius, 
                      (255, 0, 0), 2)
            
            # Add text info
            text = f"Center: {self.target_center}, Conf: {self.confidence:.2f}"
            cv2.putText(result, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw expected target size circle
            target_radius = self.target_diameter_pixels // 2
            cv2.circle(result, self.target_center, target_radius, 
                      (255, 255, 0), 2)  # Yellow circle for expected size
            
            # Draw coordinate lines
            h, w = frame.shape[:2]
            cv2.line(result, (self.target_center[0], 0), 
                    (self.target_center[0], h), (255, 0, 0), 1)
            cv2.line(result, (0, self.target_center[1]), 
                    (w, self.target_center[1]), (255, 0, 0), 1)
        
        # Show detection info
        info_text = f"Circles: {len(self.detected_circles)} | Target: {self.target_diameter_mm}mm"
        cv2.putText(result, info_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show scale info if calibrated
        if self.pixels_per_mm > 0:
            scale_text = f"Scale: {self.pixels_per_mm:.1f} px/mm"
            cv2.putText(result, scale_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def run(self):
        """
        Main detection loop.
        """
        print("Target Center Detector Started")
        print("Press 'c' to calibrate detection parameters")
        print("Press 't' to set target diameter")
        print("Press '1-9' for common target sizes (100-900mm)")
        print("Press 's' to save current frame with detections")
        print("Press 'r' to reset target center")
        print("Press 'q' to quit")
        print(f"Current target diameter: {self.target_diameter_mm}mm")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or failed to capture frame")
                break
            
            # Detect target center
            center, confidence = self.detect_target_center(frame)
            
            # Draw results
            result_frame = self.draw_detections(frame)
            
            # Display
            cv2.imshow('Target Center Detection', result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                self.calibrate_parameters()
            elif key == ord('t'):
                self.interactive_target_selection()
            elif key >= ord('1') and key <= ord('9'):
                # Quick target size selection (100-900mm)
                diameter = (key - ord('0')) * 100
                self.set_target_diameter(diameter)
                print(f"Target diameter set to {diameter}mm")
            elif key == ord('s'):
                filename = f"target_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Frame saved as {filename}")
            elif key == ord('r'):
                self.target_center = None
                self.confidence = 0.0
                print("Target center reset")
            elif key == ord('q'):
                break
        
        self.cleanup()
    
    def calibrate_parameters(self):
        """
        Interactive parameter calibration.
        """
        print("\n--- Parameter Calibration ---")
        print(f"Current parameters:")
        print(f"  Min radius: {self.min_radius}")
        print(f"  Max radius: {self.max_radius}")
        print(f"  Min distance: {self.min_dist}")
        print(f"  Param1 (edge threshold): {self.param1}")
        print(f"  Param2 (center threshold): {self.param2}")
        
        # Simple parameter adjustment
        print("Adjusting sensitivity...")
        if len(self.detected_circles) == 0:
            # No circles detected, reduce thresholds
            self.param2 = max(10, self.param2 - 5)
            self.param1 = max(30, self.param1 - 10)
            print("Reduced thresholds for better detection")
        elif len(self.detected_circles) > 10:
            # Too many circles, increase thresholds
            self.param2 = min(100, self.param2 + 5)
            self.param1 = min(100, self.param1 + 10)
            print("Increased thresholds to reduce noise")
        
        print(f"New Param1: {self.param1}, Param2: {self.param2}")
    
    def interactive_target_selection(self):
        """
        Interactive target diameter selection.
        """
        print("\n--- Target Diameter Selection ---")
        print("Common target sizes:")
        print("  1 = 100mm    2 = 200mm    3 = 300mm")
        print("  4 = 400mm    5 = 500mm    6 = 600mm")
        print("  7 = 700mm    8 = 800mm    9 = 900mm")
        print("\nOr enter custom diameter in mm (press Enter for current):")
        
        try:
            user_input = input(f"Current: {self.target_diameter_mm}mm > ")
            if user_input.strip():
                diameter = float(user_input)
                if 50 <= diameter <= 2000:  # Reasonable range
                    self.set_target_diameter(diameter)
                    print(f"Target diameter set to {diameter}mm")
                else:
                    print("Diameter must be between 50-2000mm")
            else:
                print("Keeping current diameter")
        except ValueError:
            print("Invalid input. Keeping current diameter.")
        except KeyboardInterrupt:
            print("\nCancelled.")
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nFinal Results:")
        if self.target_center is not None:
            print(f"Target center detected at: {self.target_center}")
            print(f"Detection confidence: {self.confidence:.2f}")
        else:
            print("No target center detected")


def main():
    """
    Main entry point for the application.
    """
    import sys
    
    # Default target diameter (can be overridden via command line)
    target_diameter = 200  # mm
    
    if len(sys.argv) > 1:
        try:
            target_diameter = float(sys.argv[1])
            print(f"Using command line target diameter: {target_diameter}mm")
        except ValueError:
            print(f"Invalid diameter '{sys.argv[1]}', using default {target_diameter}mm")
    
    detector = TargetCenterDetector("sample2.mp4", target_diameter_mm=target_diameter)
    detector.run()


if __name__ == "__main__":
    main()