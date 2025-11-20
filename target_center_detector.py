"""
Manual Target Center Selection with Circle Definition
Allows user to manually select target center and define concentric circles by clicking on their outer edges.
"""

import cv2
import numpy as np
from datetime import datetime
import math


class ManualTargetDetector:
    """
    Allows manual target center selection and circle definition by clicking on circle edges.
    """
    
    def __init__(self, video_source="sample2.mp4"):
        """
        Initialize the manual target selector.
        
        Args:
            video_source: Path to video file or camera index
        """
        self.cap = cv2.VideoCapture(video_source)
        self.target_center = None
        self.manual_center = None
        self.current_frame = None
        self.paused = False
        self.baseline_setup_complete = False
        
        # Circle selection
        self.circles = []  # List of circle radii
        self.selecting_circle = False
        self.current_circle_radius = 0
        
        # Mouse interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.selecting = False
        
        # Setup mouse callback
        cv2.namedWindow('Manual Target Selection')
        cv2.setMouseCallback('Manual Target Selection', self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for target center and circle edge selection.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
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
            if self.selecting:
                if not self.selecting_circle:
                    # Selecting target center
                    self.manual_center = (x, y)
                    self.target_center = (x, y)
                    self.circles = []  # Clear existing circles
                    print(f"Bullseye center set to: ({x}, {y})")
                    print("Now click on the outer edge of each target ring (start with innermost)")
                    self.selecting_circle = True
                else:
                    # Selecting circle edge
                    if self.target_center is not None:
                        radius = self.current_circle_radius
                        if radius > 5:  # Minimum radius
                            self.circles.append(radius)
                            self.circles.sort()  # Keep circles sorted by radius
                            print(f"Ring {len(self.circles)} added with radius: {radius} pixels")
                            if not self.baseline_setup_complete:
                                print("Add more rings or press 'b' to complete baseline setup")
                        else:
                            print("Ring too small, minimum radius is 5 pixels")
                
                self.selecting = False
    
    def draw_detections(self, frame):
        """
        Draw manual center, circles, and interface elements.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame
        """
        result = frame.copy()
        
        # Draw existing circles
        if self.target_center is not None and len(self.circles) > 0:
            for i, radius in enumerate(self.circles):
                # Alternate colors for better visibility
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color = colors[i % len(colors)]
                
                cv2.circle(result, self.target_center, radius, color, 2)
                
                # Add circle number
                text_pos = (self.target_center[0] + int(radius * 0.7), 
                           self.target_center[1] - int(radius * 0.7))
                cv2.putText(result, str(i+1), text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw current circle being selected
        if self.selecting_circle and self.target_center is not None and self.current_circle_radius > 0:
            cv2.circle(result, self.target_center, self.current_circle_radius, 
                      (255, 255, 255), 1)  # White preview circle
            
            # Show radius value
            text_pos = (self.mouse_x + 10, self.mouse_y - 10)
            cv2.putText(result, f"R: {self.current_circle_radius}px", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw target center
        if self.target_center is not None:
            # Draw crosshairs
            cv2.drawMarker(result, self.target_center, (0, 0, 255), 
                          cv2.MARKER_CROSS, 30, 3)
            
            # Draw coordinate lines
            h, w = frame.shape[:2]
            cv2.line(result, (self.target_center[0], 0), 
                    (self.target_center[0], h), (255, 0, 0), 1)
            cv2.line(result, (0, self.target_center[1]), 
                    (w, self.target_center[1]), (255, 0, 0), 1)
            
            # Add target center info
            text = f"Target Center: {self.target_center}"
            cv2.putText(result, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show circle count
            circle_text = f"Circles defined: {len(self.circles)}"
            cv2.putText(result, circle_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Show instruction text based on setup state
        if not self.baseline_setup_complete:
            # Prominent setup mode indicator
            cv2.rectangle(result, (5, 5), (result.shape[1]-5, 150), (0, 0, 0), -1)
            cv2.rectangle(result, (5, 5), (result.shape[1]-5, 150), (0, 255, 255), 3)
            
            if self.target_center is None:
                # Instruction text for center selection
                cv2.putText(result, "BASELINE SETUP MODE - VIDEO PAUSED", (15, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(result, "Step 1: Click on bullseye center", (15, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(result, "BASELINE SETUP MODE - VIDEO PAUSED", (15, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(result, "Step 2: Click on outer edge of each ring", (15, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                completion_text = f"Press 'b' to complete setup ({len(self.circles)} rings)"
                cv2.putText(result, completion_text, (15, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(result, "Video will start after pressing 'b'", (15, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Baseline complete status
            cv2.putText(result, "BASELINE COMPLETE - Video playing", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show circle selection instructions when in circle mode
        if self.selecting_circle and not self.baseline_setup_complete:
            instruction = "Click on ring edges (innermost to outermost)"
            cv2.putText(result, instruction, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show current mouse position
        cv2.circle(result, (self.mouse_x, self.mouse_y), 3, (255, 255, 0), -1)
        
        # Show status
        status = "PAUSED" if self.paused else "RUNNING"
        cv2.putText(result, status, (10, result.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not self.paused else (0, 0, 255), 2)
        
        return result
    
    def run(self):
        """
        Main loop with baseline setup and target/circle selection.
        """
        print("Target Baseline Setup Started")
        print("=" * 50)
        print("BASELINE SETUP MODE:")
        print("1. Video will pause on first frame")
        print("2. Click on bullseye center")
        print("3. Click on outer edge of each target ring")
        print("4. Press 'b' to complete baseline and start video")
        print("=" * 50)
        print("")
        print("Controls during setup:")
        print("  'b' - Complete baseline setup")
        print("  'c' - Clear selection and restart")
        print("  'u' - Undo last ring")
        print("  's' - Save baseline frame")
        print("  'q' - Quit")
        print("")
        print("Controls during video:")
        print("  'SPACE' - Pause/resume video")
        print("  'r' - Reset baseline (return to setup)")
        print("")
        
        while True:
            # Only read new frames after baseline setup is complete
            if not self.baseline_setup_complete:
                # During baseline setup, use only the first frame
                if self.current_frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read video file")
                        break
                    self.current_frame = frame.copy()
                    print("Video paused on first frame for baseline setup")
                    print("Click on the bullseye center to begin...")
                
                # Always use the same frame during setup
                frame = self.current_frame
            else:
                # After baseline is complete, handle normal video playback
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video")
                        break
                    self.current_frame = frame.copy()
                else:
                    # Use current frame when paused
                    frame = self.current_frame
            
            # Draw results
            result_frame = self.draw_detections(self.current_frame)
            
            # Display
            cv2.imshow('Manual Target Selection', result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('b'):
                if not self.baseline_setup_complete and self.target_center is not None:
                    self.baseline_setup_complete = True
                    self.selecting_circle = False
                    self.paused = False
                    print(f"\n" + "="*50)
                    print("BASELINE SETUP COMPLETE!")
                    print("="*50)
                    print(f"Bullseye center: {self.target_center}")
                    print(f"Target rings defined: {len(self.circles)}")
                    if len(self.circles) > 0:
                        for i, radius in enumerate(self.circles):
                            print(f"  Ring {i+1}: {radius} pixels radius")
                    print("Video playback will now begin...")
                    print("Use SPACEBAR to pause/resume during playback")
                    print("="*50 + "\n")
                elif not self.baseline_setup_complete:
                    print("ERROR: Please set bullseye center first before completing setup")
                else:
                    print("Baseline already complete")
            elif key == ord(' '):
                if self.baseline_setup_complete:
                    self.paused = not self.paused
                    print(f"Video {'paused' if self.paused else 'resumed'}")
                else:
                    print("Cannot control video playback during baseline setup")
                    print("Complete baseline setup first (press 'b' after defining target)")
            elif key == ord('r'):
                if self.baseline_setup_complete:
                    # Reset baseline - return to setup mode on first frame
                    self.baseline_setup_complete = False
                    self.target_center = None
                    self.manual_center = None
                    self.circles = []
                    self.selecting_circle = False
                    self.paused = True
                    
                    # Reset video to first frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame = frame.copy()
                    
                    print("\n" + "="*50)
                    print("BASELINE RESET")
                    print("="*50)
                    print("Returned to first frame for new baseline setup")
                    print("Click on bullseye center to begin setup...")
                    print("="*50 + "\n")
                else:
                    print("Baseline setup is not complete yet")
            elif key == ord('c'):
                self.manual_center = None
                self.target_center = None
                self.circles = []
                self.selecting_circle = False
                if not self.baseline_setup_complete:
                    print("Setup cleared - click on bullseye center")
                else:
                    print("Selection cleared")
            elif key == ord('u'):
                if len(self.circles) > 0:
                    removed = self.circles.pop()
                    print(f"Removed ring with radius {removed}px")
                else:
                    print("No rings to remove")
            elif key == ord('s'):
                if not self.baseline_setup_complete:
                    filename = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                else:
                    filename = f"target_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Frame saved as {filename}")
            elif key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nFinal Results:")
        print("=" * 40)
        if self.baseline_setup_complete:
            print("✓ Baseline setup completed")
            if self.target_center is not None:
                print(f"✓ Bullseye center: {self.target_center}")
                if len(self.circles) > 0:
                    print(f"✓ Target rings defined: {len(self.circles)}")
                    for i, radius in enumerate(self.circles):
                        diameter = radius * 2
                        print(f"    Ring {i+1}: {radius}px radius ({diameter}px diameter)")
                else:
                    print("⚠ No target rings defined")
        else:
            print("✗ Baseline setup not completed")
            if self.target_center is None:
                print("✗ No bullseye center selected")
            else:
                print(f"✓ Bullseye center: {self.target_center}")
                print(f"⚠ Partial setup - {len(self.circles)} rings defined")


def main():
    """
    Main entry point for the application.
    """
    detector = ManualTargetDetector("sample2.mp4")
    detector.run()


if __name__ == "__main__":
    main()