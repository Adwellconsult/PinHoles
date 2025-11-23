"""
Advanced Bullet Impact Detection with Stabilization and Scoring
Includes camera shake compensation and target scoring zones.
"""

import cv2
import numpy as np
from datetime import datetime
import json
import math


class AdvancedBulletImpactDetector:
    """
    Advanced detector with image stabilization, manual baseline setup, and scoring system.
    """
    
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.is_webcam = isinstance(video_source, int)
        
        # Configure webcam settings if using camera
        if self.is_webcam:
            # Set webcam properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Webcam initialized (Camera {video_source})")
            print("Live video will run until you click to start baseline setup...")
        else:
            print(f"Video file loaded: {video_source}")
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.baseline_captured = False
        self.impact_locations = []
        
        # Manual baseline setup
        self.current_frame = None
        self.baseline_setup_complete = False
        self.manual_center = None
        self.circles = []  # List of circle radii
        self.selecting_circle = False
        self.current_circle_radius = 0
        self.baseline_setup_started = False  # Track if user has started baseline setup
        
        # Debug mode
        self.debug_mode = False
        self.show_diff = False
        self.show_contours = False
        self.debug_window_open = False
        
        # Mouse interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.selecting = False
        
        # Feature detector for stabilization
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Target configuration (will be set manually)
        self.target_center = None
        self.scoring_rings = []  # Will be populated from manual setup
        
        self.total_score = 0
        
        # Setup mouse callback
        cv2.namedWindow('Advanced Detector')
        cv2.setMouseCallback('Advanced Detector', self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for target center and circle edge selection.
        """
        self.mouse_x = x
        self.mouse_y = y
        
        # Calculate current circle radius when selecting circles
        if self.selecting_circle and self.target_center is not None:
            dx = x - self.target_center[0]
            dy = y - self.target_center[1]
            self.current_circle_radius = int(math.sqrt(dx*dx + dy*dy))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecting and not self.baseline_captured:
                if not self.selecting_circle:
                    # Selecting target center
                    self.manual_center = (x, y)
                    self.target_center = (x, y)
                    self.circles = []  # Clear existing circles
                    self.baseline_setup_started = True  # Mark that baseline setup has begun
                    print(f"Bullseye center set to: ({x}, {y})")
                    print("Now click on the outer edge of each target ring")
                    self.selecting_circle = True
                else:
                    # Selecting circle edge
                    if self.target_center is not None:
                        radius = self.current_circle_radius
                        if radius > 5:  # Minimum radius
                            self.circles.append(radius)
                            self.circles.sort()  # Keep circles sorted by radius
                            print(f"Ring {len(self.circles)} added with radius: {radius} pixels")
                            print("Add more rings or press 'b' to complete baseline setup")
                        else:
                            print("Ring too small, minimum radius is 5 pixels")
                
                self.selecting = False
    
    def setup_scoring_rings(self):
        """
        Convert manually defined circles to scoring rings.
        """
        self.scoring_rings = []
        
        # Create scoring rings from manual circles
        for i, radius in enumerate(self.circles):
            # Assign decreasing point values from innermost to outermost
            points = max(10 - i, 1)  # 10, 9, 8, 7, etc., minimum 1
            ring_name = f"Ring {points}" if points < 10 else "Bullseye"
            
            self.scoring_rings.append({
                'radius': radius,
                'points': points,
                'name': ring_name
            })
    
    def create_debug_window(self, gray, diff, thresh, contours, frame_shape):
        """
        Create comprehensive debug visualization window.
        
        Args:
            gray: Grayscale frame
            diff: Difference image
            thresh: Thresholded image
            contours: Detected contours
            frame_shape: Original frame dimensions
        """
        # Resize images to fit in debug window
        height = 200
        aspect_ratio = gray.shape[1] / gray.shape[0]
        width = int(height * aspect_ratio)
        
        # Resize all debug images
        gray_resized = cv2.resize(gray, (width, height))
        diff_resized = cv2.resize(diff, (width, height))
        thresh_resized = cv2.resize(thresh, (width, height))
        
        # Create contour visualization
        contour_img = np.zeros_like(gray)
        cv2.drawContours(contour_img, contours, -1, (255), 2)
        contour_resized = cv2.resize(contour_img, (width, height))
        
        # Convert single channel images to 3-channel for display
        gray_display = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
        diff_display = cv2.cvtColor(diff_resized, cv2.COLOR_GRAY2BGR)
        thresh_display = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
        contour_display = cv2.cvtColor(contour_resized, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(gray_display, 'Grayscale', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(diff_display, 'Difference', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(thresh_display, 'Threshold', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(contour_display, f'Contours ({len(contours)})', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Create 2x2 grid layout
        top_row = np.hstack([gray_display, diff_display])
        bottom_row = np.hstack([thresh_display, contour_display])
        debug_grid = np.vstack([top_row, bottom_row])
        
        # Add overall debug info
        info_height = 100
        info_panel = np.zeros((info_height, debug_grid.shape[1], 3), dtype=np.uint8)
        
        # Add debug statistics
        cv2.putText(info_panel, f'Frame: {frame_shape[1]}x{frame_shape[0]}', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_panel, f'Contours Found: {len(contours)}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(info_panel, f'Debug Controls: t=threshold, m=morph, c=contours', (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Combine debug grid with info panel
        debug_window = np.vstack([debug_grid, info_panel])
        
        return debug_window
    
    def show_debug_info(self, contours, new_impacts):
        """
        Print detailed debug information.
        
        Args:
            contours: List of detected contours
            new_impacts: List of new impacts detected
        """
        print(f"\n--- Debug Frame Info ---")
        print(f"Total contours: {len(contours)}")
        
        valid_contours = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 20 <= area <= 5000:
                valid_contours += 1
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    print(f"  Contour {i}: area={area:.0f}, center=({cx},{cy})")
        
        print(f"Valid contours (by area): {valid_contours}")
        print(f"New impacts detected: {len(new_impacts)}")
        
        if new_impacts:
            for impact in new_impacts:
                print(f"  Impact: ({impact['x']},{impact['y']}) = {impact['score']} pts ({impact['ring']})")
        print(f"--- End Debug Info ---\n")
        
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
        """Capture baseline using manual target center and rings."""
        if self.target_center is None:
            print("ERROR: Please set target center first")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_frame = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect features for stabilization
        self.reference_keypoints, self.reference_descriptors = \
            self.orb.detectAndCompute(gray, None)
        
        # Setup scoring rings from manual circles
        self.setup_scoring_rings()
        
        self.baseline_captured = True
        self.baseline_setup_complete = True
        self.selecting_circle = False
        
        print(f"\n" + "="*50)
        print("BASELINE CAPTURED SUCCESSFULLY!")
        print("="*50)
        print(f"Target center: {self.target_center}")
        print(f"Scoring rings defined: {len(self.scoring_rings)}")
        for ring in self.scoring_rings:
            print(f"  {ring['name']}: {ring['radius']}px radius, {ring['points']} points")
        print("Impact detection now active...")
        print("="*50 + "\n")
    
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
            return [], frame
        
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
        
        # Debug visualization
        if self.debug_mode:
            debug_window = self.create_debug_window(gray, diff, thresh, contours, frame.shape)
            cv2.imshow('Debug Analysis', debug_window)
            self.debug_window_open = True
            
            # Print detailed debug info
            self.show_debug_info(contours, new_impacts)
        elif self.debug_window_open:
            cv2.destroyWindow('Debug Analysis')
            self.debug_window_open = False
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        new_impacts = []
        
        if self.debug_mode:
            print(f"Found {len(contours)} contours")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if self.debug_mode:
                print(f"Contour {i}: area = {area}")
            
            # More lenient area filtering for better detection
            if 20 <= area <= 5000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if self.debug_mode:
                        print(f"  Center: ({cx}, {cy})")
                    
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
                        
                        if self.debug_mode:
                            print(f"  NEW IMPACT: {score} points ({ring_name})")
                    elif self.debug_mode:
                        print(f"  Duplicate impact ignored")
        
        return new_impacts, stabilized
    
    def adjust_detection_parameters(self, key_pressed):
        """
        Interactively adjust detection parameters based on key presses.
        
        Args:
            key_pressed: The key that was pressed
            
        Returns:
            Updated threshold value
        """
        # This would be called from the main loop for parameter adjustment
        # For now, return default threshold
        return 25
    
    def _is_new_impact(self, x, y, min_distance=40):
        """Check if impact is new."""
        for impact in self.impact_locations:
            distance = np.sqrt((impact['x'] - x)**2 + (impact['y'] - y)**2)
            if distance < min_distance:
                return False
        return True
    
    def draw_target_and_impacts(self, frame):
        """Draw target rings, impacts, scores, and baseline setup interface."""
        display = frame.copy()
        
        # Draw manual baseline setup interface
        if not self.baseline_captured:
            # Draw existing circles during setup
            if self.target_center is not None and len(self.circles) > 0:
                for i, radius in enumerate(self.circles):
                    # Alternate colors for better visibility
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    color = colors[i % len(colors)]
                    
                    cv2.circle(display, self.target_center, radius, color, 2)
                    
                    # Add circle number
                    text_pos = (self.target_center[0] + int(radius * 0.7), 
                               self.target_center[1] - int(radius * 0.7))
                    cv2.putText(display, str(i+1), text_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw current circle being selected
            if self.selecting_circle and self.target_center is not None and self.current_circle_radius > 0:
                cv2.circle(display, self.target_center, self.current_circle_radius, 
                          (255, 255, 255), 1)  # White preview circle
                
                # Show radius value
                text_pos = (self.mouse_x + 10, self.mouse_y - 10)
                cv2.putText(display, f"R: {self.current_circle_radius}px", text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw target center
            if self.target_center is not None:
                # Draw crosshairs
                cv2.drawMarker(display, self.target_center, (0, 0, 255), 
                              cv2.MARKER_CROSS, 30, 3)
                
                # Draw coordinate lines
                h, w = frame.shape[:2]
                cv2.line(display, (self.target_center[0], 0), 
                        (self.target_center[0], h), (255, 0, 0), 1)
                cv2.line(display, (0, self.target_center[1]), 
                        (w, self.target_center[1]), (255, 0, 0), 1)
            
            # Show setup instructions
            if not self.baseline_captured:
                # Prominent setup mode indicator
                cv2.rectangle(display, (5, 5), (display.shape[1]-5, 150), (0, 0, 0), -1)
                cv2.rectangle(display, (5, 5), (display.shape[1]-5, 150), (0, 255, 255), 3)
                
                if self.target_center is None:
                    cv2.putText(display, "BASELINE SETUP - VIDEO PAUSED", (15, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display, "Step 1: Click on bullseye center", (15, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display, "BASELINE SETUP - VIDEO PAUSED", (15, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display, "Step 2: Click on outer edge of each ring", (15, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    completion_text = f"Press 'b' to complete setup ({len(self.circles)} rings)"
                    cv2.putText(display, completion_text, (15, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display, "Detection will start after pressing 'b'", (15, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return display
        
        # Draw scoring rings (after baseline is captured)
        if self.target_center and self.scoring_rings:
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
        """Main loop with manual baseline setup and impact detection."""
        print("Advanced Bullet Impact Detector with Manual Baseline")
        print("=" * 60)
        print("BASELINE SETUP MODE:")
        print("1. Video will pause on first frame (or webcam will freeze)")
        print("2. Click on bullseye center")
        print("3. Click on outer edge of each target ring")
        print("4. Press 'b' to complete baseline and start detection")
        print("=" * 60)
        print("")
        print("Controls during setup:")
        print("  'b' - Complete baseline setup")
        print("  'c' - Clear selection and restart")
        print("  'u' - Undo last ring")
        print("  's' - Save baseline frame")
        print("  'q' - Quit")
        print("")
        print("Controls during detection:")
        print("  'e' - Export results to JSON")
        print("  'r' - Reset detection (keep baseline)")
        print("  'd' - Toggle debug mode")
        print("  't' - Adjust detection threshold (when debug on)")
        print("  'SPACE' - Pause/resume video (webcam only pauses processing)")
        print("")
        
        self.paused = False
        
        while True:
            # Handle different modes: live webcam, baseline setup, and detection
            if not self.baseline_captured:
                # For webcam: continue live feed until user starts baseline setup
                if self.is_webcam and not self.baseline_setup_started:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read from webcam")
                        break
                    # Show live feed with instructions
                    display = frame.copy()
                    # Add live feed indicator
                    cv2.rectangle(display, (5, 5), (display.shape[1]-5, 80), (0, 100, 0), -1)
                    cv2.rectangle(display, (5, 5), (display.shape[1]-5, 80), (0, 255, 0), 3)
                    cv2.putText(display, "LIVE WEBCAM - Click to start baseline setup", (15, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, "Click anywhere on the target to begin...", (15, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Advanced Detector', display)
                else:
                    # During baseline setup, freeze frame (for both webcam and video)
                    if self.current_frame is None or (self.is_webcam and self.baseline_setup_started):
                        ret, frame = self.cap.read()
                        if not ret:
                            print("Failed to read video")
                            break
                        self.current_frame = frame.copy()
                        if not self.is_webcam:
                            print("Video paused on first frame for baseline setup")
                        else:
                            print("Webcam frame captured for baseline setup")
                        print("Click on the bullseye center to begin...")
                    
                    # Use frozen frame during setup
                    frame = self.current_frame
                    display = self.draw_target_and_impacts(frame)
                    cv2.imshow('Advanced Detector', display)
            else:
                # After baseline is complete, handle normal video playback
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Handle end of video file vs webcam differently
                        if not self.is_webcam:
                            # Video file ended
                            print("End of video reached - paused on last frame")
                            print("Press 'q' to quit or other keys to continue...")
                            self.paused = True
                            frame = self.current_frame if self.current_frame is not None else frame
                        else:
                            # Webcam error - try to reconnect
                            print("Webcam connection lost - attempting to reconnect...")
                            self.cap.release()
                            self.cap = cv2.VideoCapture(0)
                            continue
                    else:
                        self.current_frame = frame.copy()
                else:
                    frame = self.current_frame if self.current_frame is not None else frame
                
                new_impacts, stabilized = self.detect_impacts_with_scoring(frame)
                
                if new_impacts:
                    for impact in new_impacts:
                        self.impact_locations.append(impact)
                        print(f"Shot #{len(self.impact_locations)}: "
                        f"{impact['score']} points ({impact['ring']}) at ({impact['x']}, {impact['y']})")
                    
                    self.reference_frame = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    self.reference_frame = cv2.GaussianBlur(self.reference_frame, (5, 5), 0)
                
                # Store current frame for pausing (only when video is still playing)
                if not self.paused:
                    pass  # current_frame is already updated above when frame is read
                
                display = self.draw_target_and_impacts(stabilized)
                cv2.imshow('Advanced Detector', display)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('b'):
                if not self.baseline_captured and self.target_center is not None:
                    self.capture_baseline(self.current_frame)
                elif not self.baseline_captured:
                    print("ERROR: Please set bullseye center first")
                else:
                    print("Baseline already captured")
            elif key == ord('c'):
                if not self.baseline_captured:
                    self.manual_center = None
                    self.target_center = None
                    self.circles = []
                    self.selecting_circle = False
                    self.baseline_setup_started = False  # Reset baseline setup
                    print("Setup cleared - live feed resumed (webcam) or restart video selection")
                else:
                    print("Cannot clear baseline during detection mode")
            elif key == ord('u'):
                if not self.baseline_captured and len(self.circles) > 0:
                    removed = self.circles.pop()
                    print(f"Removed ring with radius {removed}px")
                elif not self.baseline_captured:
                    print("No rings to remove")
            elif key == ord(' '):
                if self.baseline_captured:
                    self.paused = not self.paused
                    print(f"Video {'paused' if self.paused else 'resumed'}")
                else:
                    print("Complete baseline setup first")
            elif key == ord('d'):
                if self.baseline_captured:
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                    if not self.debug_mode and self.debug_window_open:
                        cv2.destroyWindow('Debug Analysis')
                        self.debug_window_open = False
                    elif self.debug_mode:
                        print("Debug window will show processing steps")
                        print("Press 't' to adjust threshold, 'm' for morphology settings")
                else:
                    print("Complete baseline setup first")
            elif key == ord('t'):
                if self.baseline_captured and self.debug_mode:
                    # Interactive threshold adjustment
                    print("\nThreshold Adjustment:")
                    print("Current threshold: 25")
                    print("Use +/- keys to adjust during next detection cycle")
                    print("Press 'enter' to confirm changes")
                else:
                    print("Enable debug mode first ('d' key)")
            elif key == ord('e'):
                if self.baseline_captured:
                    self.export_results()
                else:
                    print("Complete baseline setup first")
            elif key == ord('r'):
                if self.baseline_captured:
                    self.impact_locations = []
                    self.total_score = 0
                    print("Detection results reset (baseline preserved)")
                else:
                    print("No detection results to reset")
            elif key == ord('s'):
                if not self.baseline_captured:
                    filename = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                else:
                    filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, display)
                print(f"Frame saved as {filename}")
            elif key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Default to webcam (0), change to video file path if needed
    # Examples:
    # source = 0          # Default webcam
    # source = 1          # Secondary webcam
    source = "sample2.mp4"  # Video file
    
    #source = 0  # Use webcam by default
    detector = AdvancedBulletImpactDetector(video_source=source)
    detector.run()
