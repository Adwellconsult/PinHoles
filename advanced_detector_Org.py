"""
Advanced Bullet Impact Detection with Stabilization and Scoring
Includes camera shake compensation and target scoring zones.
"""

import cv2
import numpy as np
from datetime import datetime
import json


class AdvancedBulletImpactDetector:
    """
    Advanced detector with image stabilization and scoring system.
    """
    
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.baseline_captured = False
        self.impact_locations = []
        
        # Feature detector for stabilization
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Target configuration (can be customized)
        self.target_center = None
        self.scoring_rings = [
            {'radius': 50, 'points': 10, 'name': 'Bullseye'},
            {'radius': 100, 'points': 9, 'name': 'Ring 9'},
            {'radius': 150, 'points': 8, 'name': 'Ring 8'},
            {'radius': 200, 'points': 7, 'name': 'Ring 7'},
            {'radius': 250, 'points': 6, 'name': 'Ring 6'},
        ]
        
        self.total_score = 0
        
    def stabilize_frame(self, frame):
        """
        Stabilize frame using feature matching with reference frame.
        
        Args:
            frame: Current frame to stabilize
            
        Returns:
            Stabilized frame aligned with reference
        """
        if self.reference_keypoints is None:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            return frame
        
        # Match features
        matches = self.bf_matcher.match(self.reference_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use top matches
        good_matches = matches[:min(50, len(matches))]
        
        if len(good_matches) < 10:
            return frame
        
        # Extract matched points
        src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt 
                              for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt 
                              for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is not None:
            h, w = frame.shape[:2]
            stabilized = cv2.warpPerspective(frame, H, (w, h))
            return stabilized
        
        return frame
    
    def capture_baseline(self, frame):
        """Capture baseline and extract features for stabilization."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_frame = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect features for stabilization
        self.reference_keypoints, self.reference_descriptors = \
            self.orb.detectAndCompute(gray, None)
        
        self.baseline_captured = True
        
        # Auto-detect target center (assume center of frame for now)
        h, w = frame.shape[:2]
        self.target_center = (w // 2, h // 2)
        
        print("Baseline captured with stabilization enabled.")
    
    def calculate_score(self, x, y):
        """
        Calculate score based on impact position relative to target center.
        
        Args:
            x, y: Impact coordinates
            
        Returns:
            Score value and ring name
        """
        if self.target_center is None:
            return 0, "Unknown"
        
        cx, cy = self.target_center
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        for ring in self.scoring_rings:
            if distance <= ring['radius']:
                return ring['points'], ring['name']
        
        return 0, "Miss"
    
    def detect_impacts_with_scoring(self, frame):
        """Detect impacts and calculate scores."""
        if not self.baseline_captured:
            return []
        
        # Stabilize frame
        stabilized = self.stabilize_frame(frame)
        
        # Process frame
        gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Frame difference
        diff = cv2.absdiff(self.reference_frame, blurred)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        new_impacts = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 50 <= area <= 3000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if self._is_new_impact(cx, cy):
                        score, ring_name = self.calculate_score(cx, cy)
                        
                        impact = {
                            'x': cx,
                            'y': cy,
                            'area': area,
                            'score': score,
                            'ring': ring_name,
                            'timestamp': datetime.now()
                        }
                        
                        new_impacts.append(impact)
                        self.total_score += score
        
        return new_impacts, stabilized
    
    def _is_new_impact(self, x, y, min_distance=40):
        """Check if impact is new."""
        for impact in self.impact_locations:
            distance = np.sqrt((impact['x'] - x)**2 + (impact['y'] - y)**2)
            if distance < min_distance:
                return False
        return True
    
    def draw_target_and_impacts(self, frame):
        """Draw target rings, impacts, and scores."""
        display = frame.copy()
        
        # Draw scoring rings
        if self.target_center:
            for ring in self.scoring_rings:
                color = (200, 200, 200) if ring['points'] != 10 else (100, 100, 255)
                cv2.circle(display, self.target_center, ring['radius'], color, 2)
                
                # Label rings
                label_pos = (self.target_center[0] + ring['radius'] - 30, 
                            self.target_center[1] - 5)
                cv2.putText(display, str(ring['points']), label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw impacts
        for i, impact in enumerate(self.impact_locations, 1):
            x, y = impact['x'], impact['y']
            score = impact['score']
            
            # Color code by score
            if score >= 9:
                color = (0, 255, 0)  # Green for high scores
            elif score >= 7:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 0, 255)  # Red for low scores
            
            cv2.circle(display, (x, y), 8, color, 2)
            cv2.circle(display, (x, y), 2, (255, 255, 255), -1)
            
            # Label with shot number and score
            cv2.putText(display, f"#{i}:{score}", (x + 12, y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display statistics
        cv2.putText(display, f"Shots: {len(self.impact_locations)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Total Score: {self.total_score}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(self.impact_locations) > 0:
            avg_score = self.total_score / len(self.impact_locations)
            cv2.putText(display, f"Average: {avg_score:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display
    
    def export_results(self, filename="shooting_results.json"):
        """Export results to JSON file."""
        results = {
            'total_shots': len(self.impact_locations),
            'total_score': self.total_score,
            'average_score': self.total_score / len(self.impact_locations) if self.impact_locations else 0,
            'shots': []
        }
        
        for i, impact in enumerate(self.impact_locations, 1):
            results['shots'].append({
                'shot_number': i,
                'x': impact['x'],
                'y': impact['y'],
                'score': impact['score'],
                'ring': impact['ring'],
                'timestamp': impact['timestamp'].isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")
    
    def run(self):
        """Main loop."""
        print("Advanced Bullet Impact Detector")
        print("Press 'b' - Capture baseline")
        print("Press 'e' - Export results to JSON")
        print("Press 'r' - Reset")
        print("Press 'q' - Quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if not self.baseline_captured:
                cv2.putText(frame, "Press 'b' for baseline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Advanced Detector', frame)
            else:
                new_impacts, stabilized = self.detect_impacts_with_scoring(frame)
                
                if new_impacts:
                    for impact in new_impacts:
                        self.impact_locations.append(impact)
                        print(f"Shot #{len(self.impact_locations)}: "
                        f"{impact['score']} points ({impact['ring']})")
                    
                    self.reference_frame = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    self.reference_frame = cv2.GaussianBlur(self.reference_frame, (5, 5), 0)
                
                display = self.draw_target_and_impacts(stabilized)
                cv2.imshow('Advanced Detector', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('b'):
                self.capture_baseline(frame)
            elif key == ord('e'):
                self.export_results()
            elif key == ord('r'):
                self.impact_locations = []
                self.total_score = 0
                self.baseline_captured = False
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    source = "sample2.mp4"  # Change to video file path if needed""
    detector = AdvancedBulletImpactDetector(video_source=source)
    detector.run()
