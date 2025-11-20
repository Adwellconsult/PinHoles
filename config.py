"""
Configuration file for bullet impact detection system.
Modify these parameters to tune detection for your specific setup.
"""

# Camera Configuration
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file like "test_video.mp4"
CAMERA_RESOLUTION = (1920, 1080)  # Set to None for default resolution

# Detection Parameters
MIN_HOLE_AREA = 50  # Minimum area in pixels for valid bullet hole
MAX_HOLE_AREA = 2000  # Maximum area in pixels for valid bullet hole
DIFF_THRESHOLD = 30  # Sensitivity for frame difference (lower = more sensitive)
MIN_CIRCULARITY = 0.3  # Minimum circularity (0-1, higher = more circular required)
MIN_IMPACT_DISTANCE = 30  # Minimum distance between impacts in pixels

# Image Processing
BLUR_KERNEL_SIZE = (5, 5)  # Gaussian blur kernel size
MORPH_KERNEL_SIZE = (3, 3)  # Morphological operation kernel size
DETECTION_COOLDOWN_FRAMES = 10  # Frames to wait before detecting again

# Stabilization (Advanced Detector)
ENABLE_STABILIZATION = True
ORB_FEATURES = 500  # Number of features for ORB detector
MIN_MATCH_COUNT = 10  # Minimum matches for homography

# Target Configuration
# Set to None for auto-detection (center of frame)
TARGET_CENTER = None  # Or set as tuple: (x, y)

# Scoring Rings (radius in pixels, points)
SCORING_RINGS = [
    {'radius': 50, 'points': 10, 'name': 'X-Ring'},
    {'radius': 100, 'points': 9, 'name': '9-Ring'},
    {'radius': 150, 'points': 8, 'name': '8-Ring'},
    {'radius': 200, 'points': 7, 'name': '7-Ring'},
    {'radius': 250, 'points': 6, 'name': '6-Ring'},
    {'radius': 300, 'points': 5, 'name': '5-Ring'},
]

# Display Settings
WINDOW_NAME = "Bullet Impact Detector"
IMPACT_CIRCLE_RADIUS = 10  # Radius for impact marker circle
IMPACT_CIRCLE_THICKNESS = 2
SHOW_FPS = True
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Colors (BGR format)
COLOR_HIGH_SCORE = (0, 255, 0)  # Green for scores >= 9
COLOR_MID_SCORE = (0, 255, 255)  # Yellow for scores 7-8
COLOR_LOW_SCORE = (0, 0, 255)  # Red for scores < 7
COLOR_RING = (200, 200, 200)  # Gray for scoring rings
COLOR_BULLSEYE = (100, 100, 255)  # Light red for bullseye

# Logging and Export
ENABLE_LOGGING = True
LOG_FILE = "impact_log.txt"
AUTO_SAVE_FRAMES = False  # Automatically save frame on each detection
FRAME_SAVE_PATH = "saved_frames/"
EXPORT_FORMAT = "json"  # "json" or "csv"

# Performance
FRAME_SKIP = 0  # Process every Nth frame (0 = process all frames)
MAX_BUFFER_SIZE = 5  # Frame buffer size for temporal filtering

# Debug Mode
DEBUG_MODE = False  # Show additional debug visualizations
SHOW_THRESHOLD = False  # Show threshold visualization in separate window
SHOW_DIFF = False  # Show frame difference in separate window
