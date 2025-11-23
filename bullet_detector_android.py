"""
Kivy-based Bullet Impact Detection App for Android
Complete port of advanced_detector.py functionality optimized for mobile devices
"""

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color, Ellipse, Line
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.storage.jsonstore import JsonStore
from kivy.utils import platform

import cv2
import numpy as np
from datetime import datetime
import json
import math
import threading
import time

kivy.require('2.0.0')

# Configuration Constants
class Config:
    """Application configuration constants"""
    # Video settings
    DEFAULT_FPS = 30.0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Detection settings
    DEFAULT_THRESHOLD = 30
    MIN_IMPACT_AREA = 15
    MAX_IMPACT_AREA = 3000
    MIN_IMPACT_DISTANCE = 30
    
    # Zoom settings
    MIN_ZOOM = 1.0
    MAX_ZOOM = 4.0
    ZOOM_STEP = 0.2
    
    # UI dimensions
    BUTTON_HEIGHT = 60
    STATS_HEIGHT = 40
    CONTROL_HEIGHT = 50
    TITLE_HEIGHT = 80
    
    # Colors (as tuples for Kivy)
    RED = (1, 0, 0, 0.8)
    GREEN = (0, 1, 0, 0.8)
    ORANGE = (1, 0.647, 0, 0.8)
    GRAY = (0.5, 0.5, 0.5, 0.8)
    
    # File filters
    VIDEO_FILTERS = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']


class UIHelpers:
    """Helper methods for UI creation to reduce duplication"""
    
    @staticmethod
    def create_button(text, size_hint_x=None, size_hint_y=None, height=Config.BUTTON_HEIGHT):
        """Create a standard button with consistent styling"""
        kwargs = {'text': text}
        if size_hint_x is not None:
            kwargs['size_hint_x'] = size_hint_x
        if size_hint_y is not None:
            kwargs['size_hint_y'] = size_hint_y
        if height is not None:
            kwargs['height'] = height
            if size_hint_y is None:
                kwargs['size_hint_y'] = None
        return Button(**kwargs)
    
    @staticmethod
    def create_label(text, size_hint_x=None, size_hint_y=None, height=None):
        """Create a standard label with consistent styling"""
        kwargs = {'text': text}
        if size_hint_x is not None:
            kwargs['size_hint_x'] = size_hint_x
        if size_hint_y is not None:
            kwargs['size_hint_y'] = size_hint_y
        if height is not None:
            kwargs['height'] = height
            if size_hint_y is None:
                kwargs['size_hint_y'] = None
        return Label(**kwargs)
    
    @staticmethod
    def create_horizontal_layout(size_hint_y=None, height=None, spacing=5):
        """Create a horizontal layout with standard settings"""
        kwargs = {'orientation': 'horizontal', 'spacing': spacing}
        if size_hint_y is not None:
            kwargs['size_hint_y'] = size_hint_y
        if height is not None:
            kwargs['height'] = height
        return BoxLayout(**kwargs)
    
    @staticmethod
    def create_vertical_layout(padding=10, spacing=5):
        """Create a vertical layout with standard settings"""
        return BoxLayout(orientation='vertical', padding=padding, spacing=spacing)


class SettingsManager:
    """Centralized settings management"""
    
    def __init__(self):
        self.store = JsonStore('app_settings.json')
    
    def get_video_source_type(self):
        """Get configured video source type"""
        try:
            return self.store.get('video_source')['type']
        except KeyError:
            return 'camera'  # default
    
    def get_video_file_path(self):
        """Get configured video file path"""
        try:
            return self.store.get('video_file_path')['path']
        except KeyError:
            return None
    
    def get_detection_threshold(self):
        """Get detection threshold setting"""
        try:
            return self.store.get('detection_threshold')['value']
        except KeyError:
            return Config.DEFAULT_THRESHOLD
    
    def get_debug_mode(self):
        """Get debug mode setting"""
        try:
            return self.store.get('debug_mode')['enabled']
        except KeyError:
            return False
    
    def save_setting(self, key, value):
        """Save a setting value"""
        self.store.put(key, **value)
    
    def get_video_source(self):
        """Get the appropriate video source based on settings"""
        source_type = self.get_video_source_type()
        if source_type == 'camera':
            return 0
        elif source_type == 'video_file':
            path = self.get_video_file_path()
            return path if path else 0
        return 0


class CameraWidget(FloatLayout):
    """
    Custom camera widget that handles video capture and OpenCV processing
    """
    
    def __init__(self, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)
        self.capture = None
        self.is_playing = False
        self.current_frame = None
        self.video_source = None
        self.is_video_file = False
        self.baseline_setup_mode = False
        self.baseline_complete = False
        self.first_frame_captured = False
        
        # Cursor tracking for target selection
        self.show_cursor = False
        self.cursor_pos = (0, 0)
        self.cursor_type = 'crosshair'  # 'crosshair' for bullseye, 'pointer' for rings
        
        # Zoom functionality for baseline setup
        self.zoom_enabled = False
        self.zoom_factor = Config.MIN_ZOOM
        self.min_zoom = Config.MIN_ZOOM
        self.max_zoom = Config.MAX_ZOOM
        self.zoom_center = (0, 0)  # Center point for zoom
        self.pan_offset = (0, 0)   # Pan offset when zoomed
        self.last_touch_pos = None
        self.is_panning = False
        
        # Detection engine
        self.detector = None
        
        # Create initial texture (will be recreated with proper size)
        self.texture = None
        self.video_rect_pos = (0, 0)
        self.video_rect_size = (0, 0)
        self.base_video_size = (0, 0)  # For aspect ratio calculations
        
        # Display rectangle
        with self.canvas:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        
        self.bind(size=self.update_rect, pos=self.update_rect)
    
    def update_rect(self, *args):
        """Update rectangle position and size maintaining aspect ratio"""
        if self.texture:
            # Recalculate aspect ratio positioning when widget size changes
            self.update_video_display()
            # Force redraw if in baseline setup mode to update interface
            if self.baseline_setup_mode and not self.baseline_complete:
                self.force_redraw()
        else:
            # Just update background rectangle
            self.rect.pos = self.pos
            self.rect.size = self.size
    
    def start_camera(self, source=0):
        """Start camera or video file capture with improved error handling"""
        try:
            self.video_source = source
            self.is_video_file = isinstance(source, str)
            
            success = self._initialize_capture(source)
            if not success:
                return False
            
            self._configure_capture()
            self._start_frame_updates()
            
            source_type = "video file" if self.is_video_file else "camera"
            Logger.info(f"CameraWidget: {source_type} started successfully")
            return True
            
        except Exception as e:
            Logger.error(f"CameraWidget: Error starting capture: {e}")
            return False
    
    def _initialize_capture(self, source):
        """Initialize video capture with error checking"""
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            source_type = "video file" if self.is_video_file else "camera"
            Logger.error(f"CameraWidget: Failed to open {source_type}")
            return False
        return True
    
    def _configure_capture(self):
        """Configure capture properties based on source type"""
        if not self.is_video_file:
            # Set camera properties for mobile devices
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            self.capture.set(cv2.CAP_PROP_FPS, Config.DEFAULT_FPS)
        else:
            # Get video file properties
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            if fps == 0:  # Some video files don't report FPS correctly
                fps = Config.DEFAULT_FPS
            Logger.info(f"CameraWidget: Video file FPS: {fps}")
    
    def _start_frame_updates(self):
        """Start the frame update timer"""
        self.is_playing = True
        Clock.schedule_interval(self.update_frame, 1.0 / Config.DEFAULT_FPS)
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_playing = False
        Clock.unschedule(self.update_frame)
        if self.capture:
            self.capture.release()
    
    def update_frame(self, dt):
        """Update frame from camera or video file"""
        if not self.capture or not self.is_playing:
            return False
        
        # If in baseline setup mode and baseline not complete, pause after first frame
        if self.baseline_setup_mode and not self.baseline_complete:
            if not self.first_frame_captured:
                # Capture and display first frame
                ret, frame = self.capture.read()
                if ret:
                    self.first_frame_captured = True
                    self.current_frame = frame
                    self.display_frame(frame)
                    
                    # Process frame with detector if available
                    if self.detector:
                        processed_frame = self.detector.process_frame(frame)
                        if processed_frame is not None:
                            self.display_frame(processed_frame)
                return True  # Keep the timer running but don't advance frame
            else:
                # Stay on the same frame until baseline is complete
                return True
        
        # Normal video progression when baseline is complete or not in setup mode
        ret, frame = self.capture.read()
        if not ret:
            if self.is_video_file:
                # End of video file - stop instead of looping
                Logger.info("CameraWidget: Video file ended - stopping playback")
                return False
            else:
                return False
        
        self.current_frame = frame.copy()
        
        # Process frame through detector if available
        if self.detector:
            processed_frame = self.detector.process_frame(frame)
            self.display_frame(processed_frame)
        else:
            self.display_frame(frame)
        
        return True
    
    def display_frame(self, frame):
        """Display frame with proper aspect ratio and zoom support"""
        if frame is None:
            return
            
        frame_h, frame_w = frame.shape[:2]
        widget_w, widget_h = self.size
        
        if widget_w == 0 or widget_h == 0:
            return
        
        # Calculate display dimensions and positioning
        display_info = self._calculate_display_dimensions(frame_w, frame_h, widget_w, widget_h)
        
        # Store for coordinate mapping
        self.video_rect_pos = display_info['rect_pos']
        self.video_rect_size = display_info['rect_size']
        self.base_video_size = display_info['base_size']
        
        # Process frame for display
        processed_frame = self._process_frame_for_display(frame, frame_w, frame_h, 
                                                        display_info['display_w'], display_info['display_h'])
        
        # Update texture and display
        self._update_texture_and_display(processed_frame, display_info['display_w'], display_info['display_h'])
    
    def _calculate_display_dimensions(self, frame_w, frame_h, widget_w, widget_h):
        """Calculate display dimensions, positioning, and scaling factors"""
        # Calculate scaling to fit frame in widget while preserving aspect ratio
        scale_w = widget_w / frame_w
        scale_h = widget_h / frame_h
        base_scale = min(scale_w, scale_h)
        
        # Calculate base display dimensions (without zoom)
        base_display_w = int(frame_w * base_scale)
        base_display_h = int(frame_h * base_scale)
        
        # Apply zoom factor
        display_w = int(base_display_w * self.zoom_factor)
        display_h = int(base_display_h * self.zoom_factor)
        
        # Calculate centering offsets
        if self.zoom_factor > 1.0:
            pan_x, pan_y = self.pan_offset
            scaled_pan_x = pan_x * display_w
            scaled_pan_y = pan_y * display_h
            offset_x = (widget_w - display_w) // 2 + int(scaled_pan_x)
            offset_y = (widget_h - display_h) // 2 + int(scaled_pan_y)
        else:
            offset_x = (widget_w - display_w) // 2
            offset_y = (widget_h - display_h) // 2
        
        return {
            'display_w': display_w,
            'display_h': display_h,
            'rect_pos': (self.pos[0] + offset_x, self.pos[1] + offset_y),
            'rect_size': (display_w, display_h),
            'base_size': (base_display_w, base_display_h)
        }
    
    def _process_frame_for_display(self, frame, frame_w, frame_h, display_w, display_h):
        """Process frame with zoom/crop and convert for display"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.zoom_factor > 1.0:
            frame_cropped = self._apply_zoom_crop(frame_rgb, frame_w, frame_h)
            frame_resized = cv2.resize(frame_cropped, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = cv2.resize(frame_rgb, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        
        return cv2.flip(frame_resized, 0)
    
    def _apply_zoom_crop(self, frame_rgb, frame_w, frame_h):
        """Apply zoom by cropping the frame"""
        crop_scale = 1.0 / self.zoom_factor
        crop_w = int(frame_w * crop_scale)
        crop_h = int(frame_h * crop_scale)
        
        # Calculate center with pan offset
        center_x = frame_w // 2 + int(self.pan_offset[0] * frame_w * 0.5)
        center_y = frame_h // 2 + int(self.pan_offset[1] * frame_h * 0.5)
        
        # Calculate crop boundaries
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(frame_w, x1 + crop_w)
        y1 = max(0, center_y - crop_h // 2)
        y2 = min(frame_h, y1 + crop_h)
        
        return frame_rgb[y1:y2, x1:x2]
    
    def _update_texture_and_display(self, processed_frame, display_w, display_h):
        """Update texture with processed frame and refresh display"""
        if (self.texture is None or 
            self.texture.width != display_w or 
            self.texture.height != display_h):
            self.texture = Texture.create(size=(display_w, display_h))
        
        buf = processed_frame.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.update_video_display()
    
    def update_video_display(self):
        """Update the video display on canvas"""
        with self.canvas:
            self.canvas.clear()
            # Black background
            Color(0, 0, 0, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            # Video texture if available
            if self.texture:
                Color(1, 1, 1, 1)
                Rectangle(texture=self.texture, pos=self.video_rect_pos, size=self.video_rect_size)
            
            # Draw paused indicator if in baseline setup mode and not complete
            if self.baseline_setup_mode and not self.baseline_complete:
                Color(1, 0, 0, 0.8)  # Red overlay
                video_x, video_y = self.video_rect_pos if self.video_rect_pos else self.pos
                video_w, video_h = self.video_rect_size if self.video_rect_size[0] else self.size
                
                # Draw pause icon (two vertical bars)
                bar_width = min(video_w * 0.02, 10)
                bar_height = min(video_h * 0.1, 30)
                center_x = video_x + video_w // 2
                center_y = video_y + video_h - 50
                
                Rectangle(pos=(center_x - 15, center_y), size=(bar_width, bar_height))
                Rectangle(pos=(center_x + 5, center_y), size=(bar_width, bar_height))
            
            # Draw target rings if detector has ring data
            self.draw_target_rings()
    
    def draw_target_rings(self):
        """Draw target center and rings overlay - only during setup phase"""
        if not self.detector:
            return
        
        ring_data = self.detector.get_ring_visualization_data()
        if not ring_data:
            return
        
        center = ring_data['center']
        rings = ring_data['rings']
        selecting = ring_data['selecting']
        
        # Only show rings and center during baseline setup (not after completion)
        if not self.detector.baseline_setup_complete:
            # Convert frame coordinates to display coordinates
            display_center = self.frame_to_display_coords(center)
            if not display_center:
                return
            
            display_x, display_y = display_center
            
            # Draw center point
            Color(1, 0, 0, 1)  # Red center
            d = 8  # Center point size
            Ellipse(pos=(display_x - d//2, display_y - d//2), size=(d, d))
            
            # Draw rings - only during setup
            for radius, ring_num in rings:
                display_radius = self.frame_to_display_scale(radius)
                if display_radius > 2:  # Only draw if visible
                    # Ring color: green for completed rings
                    Color(0, 1, 0, 0.7)  # Green for completed rings
                    
                    # Draw circle outline
                    Line(circle=(display_x, display_y, display_radius), width=2)
            
            # Draw current selection preview if selecting
            if selecting and 'current_radius' in ring_data and ring_data['current_radius'] > 10:
                preview_radius = self.frame_to_display_scale(ring_data['current_radius'])
                if preview_radius > 2:
                    Color(1, 1, 0, 0.5)  # Yellow preview
                    Line(circle=(display_x, display_y, preview_radius), width=1)
            
            # Draw crosshair at center
            Color(1, 0, 0, 1)  # Red crosshair
            Line(points=[display_x - 15, display_y, display_x + 15, display_y], width=2)
            Line(points=[display_x, display_y - 15, display_x, display_y + 15], width=2)
        
        # Draw cursor based on selection phase
        if self.show_cursor:
            cursor_x, cursor_y = self.cursor_pos
            
            if self.cursor_type == 'crosshair' and self.detector and not self.detector.target_center:
                # Phase 1: Crosshair cursor for bullseye selection
                Color(1, 0, 0, 0.8)  # Red crosshair
                cross_size = 12
                Line(points=[cursor_x - cross_size, cursor_y, cursor_x + cross_size, cursor_y], width=3)
                Line(points=[cursor_x, cursor_y - cross_size, cursor_x, cursor_y + cross_size], width=3)
                
                # Add center dot for precision
                Color(1, 0, 0, 1.0)
                Ellipse(pos=(cursor_x - 2, cursor_y - 2), size=(4, 4))
                
            elif self.cursor_type == 'pointer' and self.detector and self.detector.target_center:
                # Phase 2: Pointer cursor for ring selection
                Color(0, 1, 0, 0.8)  # Green pointer
                
                # Draw pointer arrow shape
                pointer_size = 8
                # Arrow pointing to cursor position
                Line(points=[
                    cursor_x, cursor_y,
                    cursor_x - pointer_size, cursor_y - pointer_size*2,
                    cursor_x - pointer_size//2, cursor_y - pointer_size*2,
                    cursor_x - pointer_size//2, cursor_y - pointer_size*3,
                    cursor_x + pointer_size//2, cursor_y - pointer_size*3,
                    cursor_x + pointer_size//2, cursor_y - pointer_size*2,
                    cursor_x + pointer_size, cursor_y - pointer_size*2,
                    cursor_x, cursor_y
                ], width=2)
    
    def frame_to_display_coords(self, frame_coords):
        """Convert frame coordinates to display coordinates"""
        if self.current_frame is None or not self.video_rect_size[0]:
            return None
        
        frame_x, frame_y = frame_coords
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Convert to relative coordinates
        rel_x = frame_x / frame_w
        rel_y = frame_y / frame_h
        
        # Convert to display coordinates
        video_x, video_y = self.video_rect_pos
        video_w, video_h = self.video_rect_size
        
        display_x = video_x + rel_x * video_w
        display_y = video_y + (1.0 - rel_y) * video_h  # Flip Y
        
        return (display_x, display_y)
    
    def frame_to_display_scale(self, frame_distance):
        """Convert frame distance to display distance"""
        if self.current_frame is None or not self.video_rect_size[0]:
            return 0
        
        frame_w = self.current_frame.shape[1]
        video_w = self.video_rect_size[0]
        
        scale = video_w / frame_w
        return frame_distance * scale
    
    def on_touch_down(self, touch):
        """Handle touch events for target selection and zoom"""
        if self.detector and self.collide_point(*touch.pos):
            # Check if touch is within the video display area
            video_x, video_y = self.video_rect_pos
            video_w, video_h = self.video_rect_size
            
            if (video_x <= touch.pos[0] <= video_x + video_w and
                video_y <= touch.pos[1] <= video_y + video_h):
                
                # Store touch for potential panning
                self.last_touch_pos = touch.pos
                
                # Double tap to zoom (if zoom enabled)
                if self.zoom_enabled and touch.is_double_tap:
                    if self.zoom_factor > 1.0:
                        self.reset_zoom()
                    else:
                        self.zoom_in(touch.pos)
                    return True
                
                # Show appropriate cursor based on selection phase
                if self.baseline_setup_mode:
                    self.show_cursor = True
                    self.cursor_pos = touch.pos
                    
                    if not self.detector.target_center:
                        # Phase 1: Bullseye selection - show crosshair
                        self.cursor_type = 'crosshair'
                    else:
                        # Phase 2: Ring selection - show pointer
                        self.cursor_type = 'pointer'
                
                # Convert touch coordinates to frame coordinates
                frame_coords = self.touch_to_frame_coords(touch.pos)
                if frame_coords:
                    frame_x, frame_y = frame_coords
                    self.detector.handle_touch(frame_x, frame_y, 'down')
                    
                    # Force immediate redraw during baseline setup for visual feedback
                    if self.baseline_setup_mode:
                        self.force_redraw()
                    
            return True
        return super(CameraWidget, self).on_touch_down(touch)
    
    def on_touch_move(self, touch):
        """Handle touch move events for cursor update and panning"""
        if self.detector and self.collide_point(*touch.pos) and self.show_cursor:
            # Check if touch is within the video display area
            video_x, video_y = self.video_rect_pos
            video_w, video_h = self.video_rect_size
            
            if (video_x <= touch.pos[0] <= video_x + video_w and
                video_y <= touch.pos[1] <= video_y + video_h):
                
                # Handle panning when zoomed in
                if self.zoom_enabled and self.zoom_factor > 1.0 and self.last_touch_pos:
                    if not self.is_panning and touch.grab_current is self:
                        # Check if movement is significant enough to start panning
                        dx = touch.pos[0] - self.last_touch_pos[0]
                        dy = touch.pos[1] - self.last_touch_pos[1]
                        if abs(dx) > 10 or abs(dy) > 10:
                            self.is_panning = True
                    
                    if self.is_panning:
                        # Pan the view
                        dx = touch.pos[0] - self.last_touch_pos[0]
                        dy = touch.pos[1] - self.last_touch_pos[1]
                        self.pan_view(dx, dy)
                        self.last_touch_pos = touch.pos
                        return True
                
                # Update cursor position and type
                self.cursor_pos = touch.pos
                
                # Update cursor type based on current phase
                if self.detector:
                    if not self.detector.target_center:
                        self.cursor_type = 'crosshair'  # Phase 1: Bullseye
                    else:
                        self.cursor_type = 'pointer'    # Phase 2: Rings
                
                # Force immediate redraw during baseline setup for responsive feedback
                if self.baseline_setup_mode:
                    self.force_redraw()
            
            return True
        return super(CameraWidget, self).on_touch_move(touch)
    
    def touch_to_frame_coords(self, touch_pos):
        """Convert touch position to frame coordinates with proper aspect ratio and zoom handling"""
        if self.current_frame is None or not hasattr(self, 'video_rect_size') or not self.video_rect_size[0]:
            return None
        
        video_x, video_y = self.video_rect_pos
        video_w, video_h = self.video_rect_size
        
        # Check if touch is within video area
        if (touch_pos[0] < video_x or touch_pos[0] > video_x + video_w or
            touch_pos[1] < video_y or touch_pos[1] > video_y + video_h):
            return None
        
        # Convert to relative coordinates within the video display (0.0 to 1.0)
        relative_x = (touch_pos[0] - video_x) / video_w
        relative_y = (touch_pos[1] - video_y) / video_h
        
        # Convert to frame coordinates
        frame_h, frame_w = self.current_frame.shape[:2]
        
        if self.zoom_factor > 1.0:
            # When zoomed, we need to account for the crop area
            crop_scale = 1.0 / self.zoom_factor
            crop_w = frame_w * crop_scale
            crop_h = frame_h * crop_scale
            
            # Calculate the center of the crop area (with pan offset)
            center_x = frame_w // 2 + int(self.pan_offset[0] * frame_w * 0.5)
            center_y = frame_h // 2 + int(self.pan_offset[1] * frame_h * 0.5)
            
            # Calculate crop boundaries
            x1 = max(0, center_x - crop_w // 2)
            y1 = max(0, center_y - crop_h // 2)
            
            # Map touch coordinates to the crop area
            frame_x = int(x1 + relative_x * crop_w)
            frame_y = int(y1 + (1.0 - relative_y) * crop_h)  # Flip Y for OpenCV
        else:
            # No zoom, direct mapping
            frame_x = int(relative_x * frame_w)
            frame_y = int((1.0 - relative_y) * frame_h)  # Flip Y for OpenCV
        
        # Clamp to frame boundaries
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        
        return (frame_x, frame_y)
        """Set zoom level with optional center point"""
        if not self.zoom_enabled:
            return
            
        # Clamp zoom factor
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, zoom_factor))
        
        # Set zoom center if provided
        if center_pos:
            self.zoom_center = center_pos
        
        # Reset pan if zooming out to minimum zoom
        if self.zoom_factor <= Config.MIN_ZOOM:
            self.pan_offset = (0, 0)
            self.zoom_factor = Config.MIN_ZOOM
    
    def set_zoom(self, zoom_factor, center_pos=None):
        """Set zoom level with optimal center position handling"""
        # Clamp zoom factor to valid range
        old_zoom = self.zoom_factor
        zoom_factor = max(self.min_zoom, min(self.max_zoom, zoom_factor))
        
        if center_pos and self.size[0] > 0 and self.size[1] > 0:
            # Convert center position to normalized coordinates
            norm_x = center_pos[0] / self.size[0]
            norm_y = center_pos[1] / self.size[1]
            
            # Update zoom center
            self.zoom_center = (norm_x, norm_y)
            
            # Calculate pan adjustment to keep the center point stable
            if old_zoom > 1.0 and zoom_factor > 1.0:
                # Smooth transition between zoom levels
                zoom_ratio = zoom_factor / old_zoom
                self.pan_offset = (
                    self.pan_offset[0] * zoom_ratio,
                    self.pan_offset[1] * zoom_ratio
                )
            elif zoom_factor > 1.0:
                # Initial zoom - center on the specified point
                self.pan_offset = (
                    (norm_x - 0.5) * (zoom_factor - 1.0),
                    (norm_y - 0.5) * (zoom_factor - 1.0)
                )
        
        self.zoom_factor = zoom_factor
        
        # Reset pan if zoom is back to minimum
        if self.zoom_factor <= Config.MIN_ZOOM:
            self.pan_offset = (0, 0)
            self.zoom_factor = Config.MIN_ZOOM
            self.zoom_center = (0.5, 0.5)
    
    def zoom_in(self, center_pos=None):
        """Zoom in by smaller increment for precision"""
        if center_pos is None:
            center_pos = (self.size[0] // 2, self.size[1] // 2) if self.size[0] > 0 else (0, 0)
        # Use smaller increment for finer control during baseline setup
        increment = 0.25 if self.zoom_enabled else 0.5
        self.set_zoom(self.zoom_factor + increment, center_pos)
    
    def zoom_out(self, center_pos=None):
        """Zoom out by smaller increment for precision"""
        if center_pos is None:
            center_pos = (self.size[0] // 2, self.size[1] // 2) if self.size[0] > 0 else (0, 0)
        # Use smaller increment for finer control during baseline setup
        increment = 0.25 if self.zoom_enabled else 0.5
        self.set_zoom(self.zoom_factor - increment, center_pos)
    
    def reset_zoom(self):
        """Reset to default zoom level with proper centering"""
        self.zoom_factor = Config.MIN_ZOOM
        self.pan_offset = (0, 0)
        self.zoom_center = (0.5, 0.5)
    
    def pan_view(self, delta_x, delta_y):
        """Pan the view when zoomed in with proper scaling"""
        if self.zoom_factor > 1.0 and hasattr(self, 'video_rect_size'):
            video_w, video_h = self.video_rect_size
            if video_w > 0 and video_h > 0:
                # Scale delta to normalized coordinates
                norm_delta_x = delta_x / video_w
                norm_delta_y = delta_y / video_h
                
                # Apply pan with bounds checking
                max_pan = (self.zoom_factor - 1.0) * 0.5
                current_pan_x, current_pan_y = self.pan_offset
                
                new_pan_x = max(-max_pan, min(max_pan, current_pan_x + norm_delta_x))
                new_pan_y = max(-max_pan, min(max_pan, current_pan_y + norm_delta_y))
                
                self.pan_offset = (new_pan_x, new_pan_y)
            
            # Limit panning to reasonable bounds
            max_pan = 200 * self.zoom_factor
            pan_x, pan_y = self.pan_offset
            self.pan_offset = (
                max(-max_pan, min(max_pan, pan_x)),
                max(-max_pan, min(max_pan, pan_y))
            )
    
    def on_touch_up(self, touch):
        """Handle touch release events"""
        if self.detector and self.collide_point(*touch.pos):
            # Reset panning state
            self.is_panning = False
            self.last_touch_pos = None
            
            # Check if touch is within the video display area
            video_x, video_y = self.video_rect_pos
            video_w, video_h = self.video_rect_size
            
            if (video_x <= touch.pos[0] <= video_x + video_w and
                video_y <= touch.pos[1] <= video_y + video_h):
                
                # Handle cursor transitions between phases
                was_bullseye_selection = not self.detector.target_center if self.detector else False
                
                # Convert touch coordinates to frame coordinates
                frame_coords = self.touch_to_frame_coords(touch.pos)
                if frame_coords:
                    frame_x, frame_y = frame_coords
                    self.detector.handle_touch(frame_x, frame_y, 'up')
                    
                    # Check if we transitioned from bullseye to ring selection
                    if was_bullseye_selection and self.detector.target_center:
                        # Bullseye just selected - switch to pointer cursor
                        self.cursor_type = 'pointer'
                        # Keep cursor visible for ring selection
                        self.show_cursor = True
                    elif self.detector.baseline_setup_complete:
                        # Baseline complete - hide cursor
                        self.show_cursor = False
                    
            return True
        return super(CameraWidget, self).on_touch_up(touch)
    
    def force_redraw(self):
        """Force immediate redraw of current frame with latest baseline data"""
        if self.current_frame is not None and self.detector:
            # Redraw the baseline interface with current data
            if not self.detector.baseline_captured:
                display_frame = self.detector.draw_baseline_interface(self.current_frame)
            else:
                display_frame = self.detector.draw_detection_interface(self.current_frame)
            
            # Update display immediately
            self.display_frame(display_frame)
    
    def set_baseline_setup_mode(self, enabled=True):
        """Enable or disable baseline setup mode (pauses on first frame)"""
        self.baseline_setup_mode = enabled
        if enabled:
            self.baseline_complete = False
            self.first_frame_captured = False
            # Start with crosshair cursor for bullseye selection
            self.cursor_type = 'crosshair'
            # Enable zoom for precise selection
            self.zoom_enabled = True
            self.reset_zoom()
        else:
            # Hide cursor when leaving setup mode
            self.show_cursor = False
            self.cursor_type = 'crosshair'
            # Disable zoom outside baseline setup
            self.zoom_enabled = False
            self.reset_zoom()
        
    def mark_baseline_complete(self):
        """Mark baseline setup as complete and allow video progression"""
        self.baseline_complete = True
        if self.baseline_setup_mode:
            # Resume normal video progression
            Logger.info("CameraWidget: Baseline complete, resuming video progression")


class MobileBulletDetector:
    """
    Mobile-optimized bullet impact detection engine
    Port of AdvancedBulletImpactDetector for Kivy/Android
    """
    
    def __init__(self):
        # Detection state
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.baseline_captured = False
        self.impact_locations = []
        
        # Manual baseline setup
        self.current_frame = None
        self.baseline_setup_complete = False
        self.target_center = None
        self.circles = []  # List of circle radii
        self.selecting_circle = False
        self.current_circle_radius = 0
        self.baseline_setup_started = False
        
        # Touch interaction
        self.touch_x = 0
        self.touch_y = 0
        self.selecting = False
        
        # Feature detector for stabilization (reduced features for mobile)
        self.orb = cv2.ORB_create(nfeatures=200)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Scoring system
        self.scoring_rings = []
        self.total_score = 0
        
        # Debug mode
        self.debug_mode = False
        
        # Callbacks
        self.on_impact_detected = None
        self.on_baseline_complete = None
        self.on_status_update = None
        self.on_setup_complete = None  # New callback for when setup is fully complete
    
    def handle_touch(self, x, y, action):
        """Handle touch events for target selection"""
        self.touch_x = x
        self.touch_y = y
        
        # Calculate current circle radius when selecting circles
        if self.selecting_circle and self.target_center is not None:
            dx = x - self.target_center[0]
            dy = y - self.target_center[1]
            self.current_circle_radius = int(math.sqrt(dx*dx + dy*dy))
        
        if action == 'down':
            self.selecting = True
        elif action == 'up':
            if self.selecting and not self.baseline_captured:
                if not self.selecting_circle:
                    # Selecting target center
                    self.target_center = (x, y)
                    self.circles = []  # Clear existing circles
                    self.baseline_setup_started = True
                    self.selecting_circle = True
                    
                    if self.on_status_update:
                        self.on_status_update(f"Target center set: ({x}, {y})")
                        self.on_status_update("Touch ring edges to define scoring zones")
                else:
                    # Selecting circle edge
                    if self.target_center is not None:
                        radius = self.current_circle_radius
                        if radius > 10:  # Minimum radius for mobile
                            self.circles.append(radius)
                            self.circles.sort()  # Keep circles sorted by radius
                            
                            if self.on_status_update:
                                self.on_status_update(f"Ring {len(self.circles)} added: {radius}px")
                        else:
                            if self.on_status_update:
                                self.on_status_update("Ring too small, minimum radius is 10 pixels")
                
                self.selecting = False
    
    def get_ring_visualization_data(self):
        """Get data for drawing target rings on screen"""
        if not self.target_center:
            return None
        
        # Create ring data with ring numbers for visualization
        ring_data = []
        for i, radius in enumerate(self.circles):
            ring_data.append((radius, i + 1))  # (radius, ring_number)
        
        return {
            'center': self.target_center,
            'rings': ring_data,
            'selecting': self.selecting_circle,
            'current_radius': self.current_circle_radius if self.selecting else 0
        }
    
    def complete_ring_setup(self):
        """Complete the ring selection process"""
        if self.target_center and len(self.circles) > 0:
            self.selecting_circle = False
            self.baseline_setup_complete = True
            
            # Create scoring rings from circles (highest score for smallest ring)
            self.scoring_rings = []
            for i, radius in enumerate(self.circles):
                score = len(self.circles) - i  # Smaller rings get higher scores
                self.scoring_rings.append({'radius': radius, 'score': score})
            
            if self.on_status_update:
                ring_info = ", ".join([f"Ring {len(self.circles)-i}: {radius}px" for i, radius in enumerate(self.circles)])
                self.on_status_update(f"Ring setup complete! {ring_info} - Rings now hidden.")
            
            # Trigger setup complete callback
            if self.on_setup_complete:
                self.on_setup_complete()
            
            return True
        else:
            if self.on_status_update:
                self.on_status_update("Please set target center and at least one ring before completing.")
            return False
    
    def calculate_shot_score(self, impact_point):
        """Calculate score based on impact point and defined rings"""
        if not self.target_center or not self.scoring_rings:
            return 0, "No target defined"
        
        center_x, center_y = self.target_center
        impact_x, impact_y = impact_point
        
        distance = math.sqrt((impact_x - center_x)**2 + (impact_y - center_y)**2)
        
        # Find which ring the shot hit (smaller rings = higher score)
        best_score = 0
        ring_hit = "Miss"
        
        for ring_data in self.scoring_rings:
            radius = ring_data['radius']
            score = ring_data['score']
            
            if distance <= radius:
                if score > best_score:
                    best_score = score
                    ring_hit = f"Ring {score} ({int(radius)}px)"
        
        return best_score, ring_hit
    
    def reset_baseline(self):
        """Reset baseline setup state"""
        self.target_center = None
        self.circles = []
        self.scoring_rings = []
        self.selecting_circle = False
        self.current_circle_radius = 0
        self.baseline_setup_started = False
        self.baseline_setup_complete = False
        self.baseline_captured = False
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.impact_locations = []
        self.total_score = 0
    
    def setup_scoring_rings(self):
        """Convert manually defined circles to scoring rings"""
        self.scoring_rings = []
        
        for i, radius in enumerate(self.circles):
            # Assign decreasing point values from innermost to outermost
            points = max(10 - i, 1)  # 10, 9, 8, 7, etc., minimum 1
            ring_name = f"Ring {points}" if points < 10 else "Bullseye"
            
            self.scoring_rings.append({
                'radius': radius,
                'points': points,
                'name': ring_name
            })
    
    def capture_baseline(self, frame):
        """Capture baseline using manual target center and rings"""
        if self.target_center is None:
            if self.on_status_update:
                self.on_status_update("ERROR: Please set target center first")
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_frame = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect features for stabilization (reduced for mobile performance)
        self.reference_keypoints, self.reference_descriptors = \
            self.orb.detectAndCompute(gray, None)
        
        # Setup scoring rings from manual circles
        self.setup_scoring_rings()
        
        self.baseline_captured = True
        self.baseline_setup_complete = True
        self.selecting_circle = False
        
        if self.on_baseline_complete:
            self.on_baseline_complete(self.target_center, self.scoring_rings)
        
        if self.on_status_update:
            self.on_status_update("BASELINE CAPTURED! Impact detection active")
        
        return True
    
    def stabilize_frame(self, frame):
        """Stabilize frame using feature matching (optimized for mobile)"""
        if self.reference_keypoints is None or len(self.reference_keypoints) < 5:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 5:
            return frame
        
        # Match features (reduced for performance)
        matches = self.bf_matcher.match(self.reference_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use fewer matches for mobile performance
        good_matches = matches[:min(20, len(matches))]
        
        if len(good_matches) < 5:
            return frame
        
        try:
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
        except Exception as e:
            Logger.warning(f"MobileBulletDetector: Stabilization failed: {e}")
        
        return frame
    
    def calculate_score(self, x, y):
        """Calculate score based on impact position"""
        if self.target_center is None:
            return 0, "Unknown"
        
        cx, cy = self.target_center
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        for ring in self.scoring_rings:
            if distance <= ring['radius']:
                return ring['points'], ring['name']
        
        return 0, "Miss"
    
    def detect_impacts_with_scoring(self, frame):
        """Detect impacts and calculate scores (mobile optimized)"""
        if not self.baseline_captured:
            return [], frame
        
        # Stabilize frame
        stabilized = self.stabilize_frame(frame)
        
        # Process frame
        gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Frame difference
        diff = cv2.absdiff(self.reference_frame, blurred)
        _, thresh = cv2.threshold(diff, Config.DEFAULT_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        new_impacts = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filtering for mobile cameras
            if Config.MIN_IMPACT_AREA <= area <= Config.MAX_IMPACT_AREA:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if self._is_new_impact(cx, cy):
                        score, ring_name = self.calculate_score(cx, cy)
                        
                        # Create impact record - all shots counted regardless of score
                        # (including misses with 0 points)
                        impact = {
                            'x': cx,
                            'y': cy,
                            'area': area,
                            'score': score,
                            'ring': ring_name,
                            'timestamp': datetime.now()
                        }
                        
                        # Add to shot count (includes misses)
                        new_impacts.append(impact)
                        self.total_score += score  # Score can be 0 for misses
                        
                        # Note: on_impact_detected callback moved to process_frame
                        # to ensure impact is in list before callback
        
        return new_impacts, stabilized
    
    def _is_new_impact(self, x, y, min_distance=Config.MIN_IMPACT_DISTANCE):
        """Check if impact is new (reduced distance for mobile)"""
        for impact in self.impact_locations:
            distance = np.sqrt((impact['x'] - x)**2 + (impact['y'] - y)**2)
            if distance < min_distance:
                return False
        return True
    
    def process_frame(self, frame):
        """Main frame processing method"""
        self.current_frame = frame.copy()
        
        # Handle baseline setup mode
        if not self.baseline_captured:
            return self.draw_baseline_interface(frame)
        
        # Detection mode
        new_impacts, stabilized = self.detect_impacts_with_scoring(frame)
        
        if new_impacts:
            # Add all detected impacts to shot count (including 0-score misses)
            initial_count = len(self.impact_locations)
            for impact in new_impacts:
                self.impact_locations.append(impact)
            
            final_count = len(self.impact_locations)
            added_count = final_count - initial_count
            
            # Verify all impacts were added
            if added_count != len(new_impacts):
                Logger.warning(f"Impact count mismatch: detected {len(new_impacts)}, added {added_count}")
            
            # Call impact detected callbacks AFTER impacts are added to list
            if self.on_impact_detected:
                for impact in new_impacts:
                    self.on_impact_detected(impact)
            
            # Update reference frame
            gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
            self.reference_frame = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return self.draw_detection_interface(stabilized)
    
    def draw_baseline_interface(self, frame):
        """Draw baseline setup interface"""
        display = frame.copy()
        
        # Draw existing circles during setup
        if self.target_center is not None and len(self.circles) > 0:
            for i, radius in enumerate(self.circles):
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                         (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color = colors[i % len(colors)]
                
                cv2.circle(display, self.target_center, radius, color, 2)
                
                # Add circle number
                text_pos = (self.target_center[0] + int(radius * 0.7), 
                           self.target_center[1] - int(radius * 0.7))
                cv2.putText(display, str(i+1), text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current circle being selected
        if (self.selecting_circle and self.target_center is not None 
            and self.current_circle_radius > 0):
            cv2.circle(display, self.target_center, self.current_circle_radius, 
                      (255, 255, 255), 2)
        
        # Draw target center
        if self.target_center is not None:
            cv2.drawMarker(display, self.target_center, (0, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)
        
        return display
    
    def draw_detection_interface(self, frame):
        """Draw detection interface with impacts and scoring"""
        display = frame.copy()
        
        # Draw scoring rings
        if self.target_center and self.scoring_rings:
            for ring in self.scoring_rings:
                color = (200, 200, 200) if ring['points'] != 10 else (100, 100, 255)
                cv2.circle(display, self.target_center, ring['radius'], color, 2)
                
                # Label rings
                label_pos = (self.target_center[0] + ring['radius'] - 20, 
                            self.target_center[1] - 5)
                cv2.putText(display, str(ring['points']), label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw impacts
        for i, impact in enumerate(self.impact_locations, 1):
            x, y = impact['x'], impact['y']
            score = impact['score']
            
            # Color code by score (misses = orange, hits = red/yellow/green)
            if score >= 9:
                color = (0, 255, 0)  # Green for high scores
            elif score >= 7:
                color = (0, 255, 255)  # Yellow for medium scores
            elif score > 0:
                color = (0, 0, 255)  # Red for low scores (hits)
            else:
                color = (0, 165, 255)  # Orange for misses (0 points)
            
            cv2.circle(display, (x, y), 6, color, 2)
            cv2.circle(display, (x, y), 2, (255, 255, 255), -1)
            
            # Label with shot number and score
            cv2.putText(display, f"#{i}:{score}", (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return display
    
    def reset_baseline(self):
        """Reset baseline setup"""
        self.baseline_captured = False
        self.baseline_setup_complete = False
        self.target_center = None
        self.circles = []
        self.selecting_circle = False
        self.baseline_setup_started = False
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
    
    def reset_detection(self):
        """Reset detection results but keep baseline"""
        self.impact_locations = []
        self.total_score = 0
    
    def get_results(self):
        """Get detection results"""
        return {
            'total_shots': len(self.impact_locations),
            'total_score': self.total_score,
            'average_score': self.total_score / len(self.impact_locations) if self.impact_locations else 0,
            'target_center': self.target_center,
            'scoring_rings': self.scoring_rings,
            'shots': [
                {
                    'shot_number': i,
                    'x': impact['x'],
                    'y': impact['y'],
                    'score': impact['score'],
                    'ring': impact['ring'],
                    'timestamp': impact['timestamp'].isoformat()
                }
                for i, impact in enumerate(self.impact_locations, 1)
            ]
        }


class BaselineSetupScreen(Screen):
    """Screen for baseline setup"""
    
    def __init__(self, **kwargs):
        super(BaselineSetupScreen, self).__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        """Build baseline setup UI"""
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Instructions
        instructions = Label(
            text="Baseline Setup (Zoom Enabled):\n"
                 "1. Touch bullseye center (crosshair cursor)\n"
                 "2. Touch ring edges (pointer cursor)\n"
                 "• Use Zoom +/- for precise targeting\n"
                 "• Double-tap or pinch to zoom\n"
                 "• Reset button returns to 1.0x zoom",
            size_hint_y=None,
            height=140,
            halign='center'
        )
        
        # Camera view
        self.camera_widget = CameraWidget()
        
        # Control buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        
        # Zoom controls (left side)
        zoom_layout = BoxLayout(orientation='horizontal', size_hint_x=0.4)
        
        self.zoom_in_button = Button(
            text='Zoom +',
            size_hint_x=0.33
        )
        self.zoom_in_button.bind(on_press=self.zoom_in)
        
        self.zoom_out_button = Button(
            text='Zoom -',
            size_hint_x=0.33
        )
        self.zoom_out_button.bind(on_press=self.zoom_out)
        
        self.zoom_reset_button = Button(
            text='Reset',
            size_hint_x=0.34
        )
        self.zoom_reset_button.bind(on_press=self.reset_zoom)
        
        zoom_layout.add_widget(self.zoom_in_button)
        zoom_layout.add_widget(self.zoom_out_button)
        zoom_layout.add_widget(self.zoom_reset_button)
        
        # Main control buttons (right side)
        control_layout = BoxLayout(orientation='horizontal', size_hint_x=0.6)
        
        self.complete_button = Button(
            text='Complete Setup',
            size_hint_x=0.4,
            disabled=True
        )
        self.complete_button.bind(on_press=self.complete_setup)
        
        self.clear_button = Button(text='Clear', size_hint_x=0.3)
        self.clear_button.bind(on_press=self.clear_setup)
        
        self.back_button = Button(text='Back', size_hint_x=0.3)
        self.back_button.bind(on_press=self.go_back)
        
        control_layout.add_widget(self.complete_button)
        control_layout.add_widget(self.clear_button)
        control_layout.add_widget(self.back_button)
        
        button_layout.add_widget(zoom_layout)
        button_layout.add_widget(control_layout)
        
        # Status label
        self.status_label = Label(
            text='Touch target center to begin',
            size_hint_y=None,
            height=40
        )
        
        # Ring count and zoom info labels
        info_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        
        self.ring_count_label = Label(
            text='Rings: 0',
            size_hint_x=0.5
        )
        
        self.zoom_info_label = Label(
            text='Zoom: 1.0x',
            size_hint_x=0.5
        )
        
        info_layout.add_widget(self.ring_count_label)
        info_layout.add_widget(self.zoom_info_label)
        
        main_layout.add_widget(instructions)
        main_layout.add_widget(self.camera_widget)
        main_layout.add_widget(button_layout)
        main_layout.add_widget(info_layout)
        main_layout.add_widget(self.status_label)
        
        self.add_widget(main_layout)
    
    def on_enter(self):
        """Called when screen is entered"""
        # Get video source from settings
        app = App.get_running_app()
        menu_screen = app.root.get_screen('menu')
        video_source = menu_screen.get_video_source()
        
        # Start camera/video and setup detector
        if self.camera_widget.start_camera(video_source):
            # Enable baseline setup mode to pause on first frame
            self.camera_widget.set_baseline_setup_mode(True)
            
            detector = MobileBulletDetector()
            detector.on_status_update = self.update_status
            detector.on_baseline_complete = self.on_baseline_complete
            detector.on_setup_complete = self.on_setup_complete  # Handle when rings are complete
            self.camera_widget.detector = detector
            
            # Initialize zoom display
            self.update_zoom_display()
            
            source_type = 'video file' if isinstance(video_source, str) else 'camera'
            self.status_label.text = f'Using {source_type} - Touch target center to begin'
        else:
            source_type = 'video file' if isinstance(video_source, str) else 'camera'
            self.status_label.text = f'Failed to start {source_type}'
    
    def on_leave(self):
        """Called when screen is left"""
        self.camera_widget.stop_camera()
    
    def update_status(self, message):
        """Update status display with enhanced zoom info"""
        self.status_label.text = message
        if self.camera_widget.detector:
            ring_count = len(self.camera_widget.detector.circles)
            self.ring_count_label.text = f'Rings: {ring_count}'
            
            # Update zoom info with more precision for small increments
            zoom_factor = self.camera_widget.zoom_factor
            if zoom_factor == Config.MIN_ZOOM:
                self.zoom_info_label.text = 'Zoom: 1.0x'
            else:
                self.zoom_info_label.text = f'Zoom: {zoom_factor:.2f}x'
            
            # Enable complete button if target center is set and setup is not yet complete
            has_target = self.camera_widget.detector.target_center is not None
            setup_complete = self.camera_widget.detector.baseline_setup_complete
            baseline_captured = self.camera_widget.detector.baseline_captured
            
            # Button enabled when: has target AND (setup complete OR baseline not captured)
            self.complete_button.disabled = not (has_target and (setup_complete or not baseline_captured))
    
    def zoom_in(self, instance):
        """Zoom in the camera view"""
        if self.camera_widget.zoom_enabled and hasattr(self.camera_widget, 'zoom_factor'):
            center_pos = (self.camera_widget.size[0] // 2, self.camera_widget.size[1] // 2)
            self.camera_widget.zoom_in(center_pos)
            self.update_zoom_display()
    
    def zoom_out(self, instance):
        """Zoom out the camera view"""
        if self.camera_widget.zoom_enabled and hasattr(self.camera_widget, 'zoom_factor'):
            center_pos = (self.camera_widget.size[0] // 2, self.camera_widget.size[1] // 2)
            self.camera_widget.zoom_out(center_pos)
            self.update_zoom_display()
    
    def reset_zoom(self, instance):
        """Reset zoom to default level"""
        self.camera_widget.reset_zoom()
        self.update_zoom_display()
    
    def update_zoom_display(self):
        """Update zoom level display with enhanced precision"""
        if hasattr(self.camera_widget, 'zoom_factor'):
            zoom_factor = self.camera_widget.zoom_factor
            if zoom_factor == Config.MIN_ZOOM:
                self.zoom_info_label.text = 'Zoom: 1.0x'
            else:
                self.zoom_info_label.text = f'Zoom: {zoom_factor:.2f}x'
            
            # Update button states with proper threshold checking
            self.zoom_in_button.disabled = zoom_factor >= (self.camera_widget.max_zoom - 0.01)
            self.zoom_out_button.disabled = zoom_factor <= (self.camera_widget.min_zoom + 0.01)
        else:
            self.zoom_info_label.text = 'Zoom: 1.0x'
            self.zoom_in_button.disabled = False
            self.zoom_out_button.disabled = True
    
    def complete_setup(self, instance):
        """Complete baseline setup"""
        if self.camera_widget.detector:
            # If ring setup isn't complete, complete it first
            if not self.camera_widget.detector.baseline_setup_complete:
                if self.camera_widget.detector.complete_ring_setup():
                    # Ring setup is now complete, video will progress
                    return
            
            # Ring setup is complete, now capture baseline frame
            if self.camera_widget.current_frame is not None:
                success = self.camera_widget.detector.capture_baseline(
                    self.camera_widget.current_frame
                )
                if success:
                    # Pass detector to detection screen
                    app = App.get_running_app()
                    detection_screen = app.root.get_screen('detection')
                    detection_screen.set_detector(self.camera_widget.detector)
                    app.root.current = 'detection'
    
    def clear_setup(self, instance):
        """Clear current setup and reset zoom for fresh start"""
        if self.camera_widget.detector:
            self.camera_widget.detector.reset_baseline()
            self.camera_widget.set_baseline_setup_mode(True)  # Re-enable pause on first frame
            # Reset zoom for fresh start
            self.camera_widget.reset_zoom()
            self.update_zoom_display()
            self.complete_button.text = 'Complete Setup'  # Reset button text
            self.update_status('Setup cleared - Touch target center to begin')
    
    def go_back(self, instance):
        """Go back to main menu"""
        app = App.get_running_app()
        app.root.current = 'menu'
    
    def on_baseline_complete(self, target_center, scoring_rings):
        """Called when baseline is complete"""
        self.status_label.text = f'Baseline complete! Rings: {len(scoring_rings)}'
    
    def on_setup_complete(self):
        """Called when ring setup is complete - allow video to progress"""
        self.camera_widget.mark_baseline_complete()
        self.status_label.text = 'Ring setup complete! Rings are now hidden. Press Complete Setup to capture baseline.'
        
        # Enable the complete button for final baseline capture
        self.complete_button.disabled = False
        self.complete_button.text = 'Complete Baseline Capture'


class DetectionScreen(Screen):
    """Screen for impact detection"""
    
    def __init__(self, **kwargs):
        super(DetectionScreen, self).__init__(**kwargs)
        self.detector = None
        self.build_ui()
    
    def build_ui(self):
        """Build detection UI"""
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=5)
        
        # Camera view
        self.camera_widget = CameraWidget()
        
        # Statistics panel
        stats_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        
        self.shots_label = Label(text='Shots: 0', size_hint_x=0.25)
        self.score_label = Label(text='Score: 0', size_hint_x=0.25)
        self.avg_label = Label(text='Avg: 0.0', size_hint_x=0.25)
        self.status_label = Label(text='Ready', size_hint_x=0.25)
        
        stats_layout.add_widget(self.shots_label)
        stats_layout.add_widget(self.score_label)
        stats_layout.add_widget(self.avg_label)
        stats_layout.add_widget(self.status_label)
        
        # Control buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        
        self.pause_button = Button(text='Pause', size_hint_x=0.2)
        self.pause_button.bind(on_press=self.toggle_pause)
        
        self.reset_button = Button(text='Reset', size_hint_x=0.2)
        self.reset_button.bind(on_press=self.reset_detection)
        
        self.export_button = Button(text='Export', size_hint_x=0.2)
        self.export_button.bind(on_press=self.export_results)
        
        self.debug_button = Button(text='Debug', size_hint_x=0.2)
        self.debug_button.bind(on_press=self.toggle_debug)
        
        self.menu_button = Button(text='Menu', size_hint_x=0.2)
        self.menu_button.bind(on_press=self.go_to_menu)
        
        button_layout.add_widget(self.pause_button)
        button_layout.add_widget(self.reset_button)
        button_layout.add_widget(self.export_button)
        button_layout.add_widget(self.debug_button)
        button_layout.add_widget(self.menu_button)
        
        main_layout.add_widget(self.camera_widget)
        main_layout.add_widget(stats_layout)
        main_layout.add_widget(button_layout)
        
        self.add_widget(main_layout)
    
    def set_detector(self, detector):
        """Set the detection engine"""
        self.detector = detector
        self.detector.on_impact_detected = self.on_impact_detected
        self.detector.on_status_update = self.update_status
        self.camera_widget.detector = detector
    
    def on_enter(self):
        """Called when screen is entered"""
        if self.detector and not self.camera_widget.is_playing:
            # Get video source from settings
            app = App.get_running_app()
            menu_screen = app.root.get_screen('menu')
            video_source = menu_screen.get_video_source()
            self.camera_widget.start_camera(video_source)
    
    def on_leave(self):
        """Called when screen is left"""
        self.camera_widget.stop_camera()
    
    def on_impact_detected(self, impact):
        """Called when new impact is detected - now called after impact is added to list"""
        # Update stats first to ensure accurate counts
        self.update_stats()
        
        # Get current shot number (should be accurate now)
        shot_number = len(self.detector.impact_locations)
        self.status_label.text = f"Shot #{shot_number}: {impact['score']} pts (Total: {shot_number})"
    
    def update_status(self, message):
        """Update status display"""
        self.status_label.text = message
    
    def update_stats(self):
        """Update statistics display with verified counts"""
        if not self.detector:
            return
        
        # Get total detected impacts (all shots including misses)
        total_impacts = len(self.detector.impact_locations)
        total_score = self.detector.total_score
        avg_score = total_score / total_impacts if total_impacts > 0 else 0
        
        # Debug: Log first shot specifically
        if total_impacts == 1:
            print(f"DEBUG: First shot detected - Total impacts: {total_impacts}")
        
        # Update display with verified total
        self.shots_label.text = f'Shots: {total_impacts}'  # All detected impacts
        self.score_label.text = f'Score: {total_score}'
        self.avg_label.text = f'Avg: {avg_score:.1f}'
    
    def toggle_pause(self, instance):
        """Toggle pause/resume"""
        if self.camera_widget.is_playing:
            self.camera_widget.is_playing = False
            self.pause_button.text = 'Resume'
        else:
            self.camera_widget.is_playing = True
            self.pause_button.text = 'Pause'
    
    def reset_detection(self, instance):
        """Reset detection results"""
        if self.detector:
            self.detector.reset_detection()
            self.update_stats()
            self.status_label.text = 'Detection reset'
    
    def export_results(self, instance):
        """Export results to JSON"""
        if not self.detector or not self.detector.impact_locations:
            self.status_label.text = 'No results to export'
            return
        
        try:
            results = self.detector.get_results()
            filename = f"shooting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save to app's data directory
            if platform == 'android':
                from android.storage import primary_external_storage_path
                storage_path = primary_external_storage_path()
                filepath = f"{storage_path}/{filename}"
            else:
                filepath = filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.status_label.text = f'Exported: {filename}'
        except Exception as e:
            Logger.error(f"Export failed: {e}")
            self.status_label.text = 'Export failed'
    
    def toggle_debug(self, instance):
        """Toggle debug mode"""
        if self.detector:
            self.detector.debug_mode = not self.detector.debug_mode
            self.debug_button.text = 'Debug ON' if self.detector.debug_mode else 'Debug'
    
    def go_to_menu(self, instance):
        """Return to main menu"""
        app = App.get_running_app()
        app.root.current = 'menu'


class MenuScreen(Screen):
    """Main menu screen"""
    
    def __init__(self, **kwargs):
        super(MenuScreen, self).__init__(**kwargs)
        # Settings storage
        self.settings_store = JsonStore('app_settings.json')
        self.build_ui()
    
    def build_ui(self):
        """Build menu UI"""
        main_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Title
        title = Label(
            text='Bullet Impact Detector',
            font_size='24sp',
            size_hint_y=None,
            height=80
        )
        
        # Menu buttons
        start_button = Button(
            text='Start New Session',
            size_hint_y=None,
            height=60
        )
        start_button.bind(on_press=self.start_session)
        
        settings_button = Button(
            text='Settings',
            size_hint_y=None,
            height=60
        )
        settings_button.bind(on_press=self.open_settings)
        
        help_button = Button(
            text='Help',
            size_hint_y=None,
            height=60
        )
        help_button.bind(on_press=self.open_help)
        
        exit_button = Button(
            text='Exit',
            size_hint_y=None,
            height=60
        )
        exit_button.bind(on_press=self.exit_app)
        
        main_layout.add_widget(title)
        main_layout.add_widget(start_button)
        main_layout.add_widget(settings_button)
        main_layout.add_widget(help_button)
        main_layout.add_widget(exit_button)
        
        self.add_widget(main_layout)
    
    def start_session(self, instance):
        """Start new detection session"""
        app = App.get_running_app()
        app.root.current = 'baseline'
    
    def open_settings(self, instance):
        """Open settings popup"""
        # Create settings popup
        settings_content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Video source selection
        source_layout = BoxLayout(orientation='vertical')
        source_layout.add_widget(Label(text='Video Source:', size_hint_y=None, height=30))
        
        source_buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        
        self.camera_button = Button(text='Camera', state='down')
        self.video_file_button = Button(text='Video File', state='normal')
        
        # Load saved preference
        try:
            current_source = self.settings_store.get('video_source')['type']
        except:
            current_source = 'camera'
        if current_source == 'video_file':
            self.camera_button.state = 'normal'
            self.video_file_button.state = 'down'
        
        self.camera_button.bind(on_press=self.select_camera_source)
        self.video_file_button.bind(on_press=self.select_video_file)
        
        source_buttons.add_widget(self.camera_button)
        source_buttons.add_widget(self.video_file_button)
        
        source_layout.add_widget(source_buttons)
        
        # Show current video file if selected
        try:
            current_file = self.settings_store.get('video_file_path')['path']
        except:
            current_file = 'No file selected'
        self.file_label = Label(
            text=f'File: {current_file}', 
            size_hint_y=None, 
            height=30,
            text_size=(400, None),
            halign='left'
        )
        source_layout.add_widget(self.file_label)
        
        # Detection threshold slider
        threshold_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        threshold_layout.add_widget(Label(text='Detection Threshold:', size_hint_x=0.6))
        try:
            current_threshold = self.settings_store.get('detection_threshold')['value']
        except:
            current_threshold = 30
        self.threshold_slider = Slider(min=10, max=50, value=current_threshold, step=1, size_hint_x=0.4)
        threshold_layout.add_widget(self.threshold_slider)
        
        # Debug mode switch
        debug_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        debug_layout.add_widget(Label(text='Debug Mode:', size_hint_x=0.6))
        try:
            current_debug = self.settings_store.get('debug_mode')['enabled']
        except:
            current_debug = False
        self.debug_switch = Switch(active=current_debug, size_hint_x=0.4)
        debug_layout.add_widget(self.debug_switch)
        
        settings_content.add_widget(source_layout)
        settings_content.add_widget(threshold_layout)
        settings_content.add_widget(debug_layout)
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        save_button = Button(text='Save', size_hint_x=0.5)
        close_button = Button(text='Close', size_hint_x=0.5)
        
        button_layout.add_widget(save_button)
        button_layout.add_widget(close_button)
        settings_content.add_widget(button_layout)
        
        self.settings_popup = Popup(title='Settings', content=settings_content, size_hint=(0.9, 0.7))
        
        save_button.bind(on_press=self.save_settings)
        close_button.bind(on_press=self.settings_popup.dismiss)
        self.settings_popup.open()
    
    def open_help(self, instance):
        """Open help popup"""
        help_text = """
        Bullet Impact Detector Help
        
        1. Baseline Setup:
           - Point camera at clean target
           - Touch the bullseye center
           - Touch the outer edge of each scoring ring
           - Press 'Complete Setup'
        
        2. Detection:
           - Fire shots at target
           - Impacts are automatically detected
           - Scores calculated based on ring zones
           - View statistics in real-time
        
        3. Results:
           - Export results to JSON file
           - Reset detection to start over
           - Keep baseline for new rounds
        
        Tips:
        - Ensure good lighting
        - Mount phone steady
        - Clean target for best results
        """
        
        help_content = BoxLayout(orientation='vertical', padding=10)
        help_label = Label(text=help_text, text_size=(None, None), halign='left', valign='top')
        close_button = Button(text='Close', size_hint_y=None, height=40)
        
        help_content.add_widget(help_label)
        help_content.add_widget(close_button)
        
        popup = Popup(title='Help', content=help_content, size_hint=(0.9, 0.8))
        close_button.bind(on_press=popup.dismiss)
        popup.open()
    
    def select_camera_source(self, instance):
        """Select camera as video source"""
        self.camera_button.state = 'down'
        self.video_file_button.state = 'normal'
    
    def select_video_file(self, instance):
        """Select video file as source"""
        self.camera_button.state = 'normal'
        self.video_file_button.state = 'down'
        self.open_file_chooser()
    
    def open_file_chooser(self):
        """Open file chooser for video selection"""
        # Create file chooser popup
        file_content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # File chooser
        filechooser = FileChooserListView(
            filters=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv'],
            size_hint_y=0.8
        )
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        select_button = Button(text='Select', size_hint_x=0.5)
        cancel_button = Button(text='Cancel', size_hint_x=0.5)
        
        button_layout.add_widget(select_button)
        button_layout.add_widget(cancel_button)
        
        file_content.add_widget(filechooser)
        file_content.add_widget(button_layout)
        
        file_popup = Popup(title='Select Video File', content=file_content, size_hint=(0.9, 0.8))
        
        def select_file(instance):
            if filechooser.selection:
                selected_file = filechooser.selection[0]
                self.file_label.text = f'File: {selected_file}'
                file_popup.dismiss()
        
        select_button.bind(on_press=select_file)
        cancel_button.bind(on_press=file_popup.dismiss)
        file_popup.open()
    
    def save_settings(self, instance):
        """Save settings to storage"""
        # Save video source type
        source_type = 'camera' if self.camera_button.state == 'down' else 'video_file'
        self.settings_store.put('video_source', type=source_type)
        
        # Save video file path if video file is selected
        if source_type == 'video_file' and 'File:' in self.file_label.text:
            file_path = self.file_label.text.replace('File: ', '')
            if file_path != 'No file selected':
                self.settings_store.put('video_file_path', path=file_path)
        
        # Save threshold
        self.settings_store.put('detection_threshold', value=int(self.threshold_slider.value))
        
        # Save debug mode
        self.settings_store.put('debug_mode', enabled=self.debug_switch.active)
        
        self.settings_popup.dismiss()
    
    def get_video_source(self):
        """Get the configured video source"""
        try:
            source_config = self.settings_store.get('video_source')
        except:
            source_config = {'type': 'camera'}
        
        if source_config['type'] == 'camera':
            return 0  # Default camera
        else:
            try:
                file_config = self.settings_store.get('video_file_path')
                return file_config['path'] if file_config['path'] else 0
            except:
                return 0
    
    def exit_app(self, instance):
        """Exit application"""
        App.get_running_app().stop()


class BulletDetectorApp(App):
    """Main application class"""
    
    def build(self):
        """Build the application"""
        # Create screen manager
        sm = ScreenManager()
        
        # Add screens
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(BaselineSetupScreen(name='baseline'))
        sm.add_widget(DetectionScreen(name='detection'))
        
        return sm
    
    def on_pause(self):
        """Handle app pause (Android)"""
        return True
    
    def on_resume(self):
        """Handle app resume (Android)"""
        pass


if __name__ == '__main__':
    BulletDetectorApp().run()