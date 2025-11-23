"""
Kivy Video Player with Enhanced Debugging
Helps identify video loading issues
"""

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.video import Video
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.logger import Logger
import os

kivy.require('2.0.0')


class DebugVideoPlayerApp(App):
    def __init__(self, **kwargs):
        super(DebugVideoPlayerApp, self).__init__(**kwargs)
        self.video = None
        self.is_playing = False
        
    def build(self):
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Video file path - try absolute path
        video_path = os.path.abspath('sample2.mp4')
        
        # Debug info
        debug_info = f"""
DEBUG INFO:
Current working directory: {os.getcwd()}
Video path: {video_path}
File exists: {os.path.exists(video_path)}
"""
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            debug_info += f"File size: {file_size} bytes\n"
        
        print(debug_info)
        Logger.info(f"VideoPlayer: {debug_info}")
        
        # Create video widget with debugging
        try:
            self.video = Video(
                source=video_path,
                state='stop',
                options={'allow_stretch': True, 'eos': 'loop'}
            )
            
            # Bind video events for debugging
            self.video.bind(state=self.on_video_state_change)
            self.video.bind(loaded=self.on_video_loaded)
            
            print(f"Video widget created successfully")
            Logger.info("VideoPlayer: Video widget created")
            
        except Exception as e:
            print(f"Error creating video widget: {e}")
            Logger.error(f"VideoPlayer: Error creating video widget: {e}")
            # Create placeholder if video fails
            self.video = Label(text=f"Video Load Error:\n{str(e)}", halign='center')
        
        # Control layout
        control_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        # Play/Pause button
        self.play_button = Button(text='Play', size_hint_x=None, width=100)
        self.play_button.bind(on_press=self.toggle_play)
        
        # Stop button
        stop_button = Button(text='Stop', size_hint_x=None, width=100)
        stop_button.bind(on_press=self.stop_video)
        
        # Reload button for debugging
        reload_button = Button(text='Reload', size_hint_x=None, width=100)
        reload_button.bind(on_press=self.reload_video)
        
        # Position slider
        self.position_slider = Slider(min=0, max=100, value=0, step=1)
        
        # Time label
        self.time_label = Label(text='00:00 / 00:00', size_hint_x=None, width=120)
        
        # Status label with debug info
        self.status_label = Label(
            text=f'Video: {video_path}\nExists: {os.path.exists(video_path)}', 
            size_hint_y=None, 
            height=60,
            halign='left',
            valign='top'
        )
        self.status_label.text_size = (None, None)
        
        # Add controls to layout
        control_layout.add_widget(self.play_button)
        control_layout.add_widget(stop_button)
        control_layout.add_widget(reload_button)
        control_layout.add_widget(self.position_slider)
        control_layout.add_widget(self.time_label)
        
        # Add all to main layout
        main_layout.add_widget(self.video)
        main_layout.add_widget(control_layout)
        main_layout.add_widget(self.status_label)
        
        # Schedule position update and debug check
        Clock.schedule_interval(self.update_debug_info, 1.0)
        
        return main_layout
    
    def on_video_state_change(self, instance, state):
        """Debug video state changes"""
        print(f"Video state changed to: {state}")
        Logger.info(f"VideoPlayer: Video state changed to: {state}")
        self.update_status()
    
    def on_video_loaded(self, instance, loaded):
        """Debug video loading"""
        print(f"Video loaded: {loaded}")
        Logger.info(f"VideoPlayer: Video loaded: {loaded}")
        if hasattr(self.video, 'duration'):
            print(f"Video duration: {self.video.duration}")
            Logger.info(f"VideoPlayer: Video duration: {self.video.duration}")
        self.update_status()
    
    def reload_video(self, instance):
        """Reload video for debugging"""
        try:
            video_path = os.path.abspath('sample2.mp4')
            print(f"Reloading video: {video_path}")
            if hasattr(self.video, 'source'):
                self.video.state = 'stop'
                self.video.source = video_path
                self.video.state = 'stop'  # Reset state
            self.update_status()
        except Exception as e:
            print(f"Error reloading video: {e}")
            Logger.error(f"VideoPlayer: Error reloading video: {e}")
    
    def toggle_play(self, instance):
        """Toggle between play and pause"""
        if not hasattr(self.video, 'state'):
            self.status_label.text = "Error: Video widget not properly initialized"
            return
            
        try:
            if self.video.state == 'play':
                self.video.state = 'pause'
                self.play_button.text = 'Play'
                self.is_playing = False
            else:
                self.video.state = 'play'
                self.play_button.text = 'Pause'
                self.is_playing = True
            self.update_status()
        except Exception as e:
            print(f"Error toggling play: {e}")
            self.status_label.text = f"Play Error: {e}"
    
    def stop_video(self, instance):
        """Stop video playback"""
        try:
            if hasattr(self.video, 'state'):
                self.video.state = 'stop'
                self.play_button.text = 'Play'
                self.is_playing = False
                self.position_slider.value = 0
            self.update_status()
        except Exception as e:
            print(f"Error stopping video: {e}")
            self.status_label.text = f"Stop Error: {e}"
    
    def update_debug_info(self, dt):
        """Update debug information"""
        if hasattr(self.video, 'state'):
            try:
                debug_text = f"State: {self.video.state}"
                if hasattr(self.video, 'duration') and self.video.duration:
                    debug_text += f" | Duration: {self.video.duration:.1f}s"
                if hasattr(self.video, 'position'):
                    debug_text += f" | Position: {self.video.position:.1f}s"
                if hasattr(self.video, 'loaded'):
                    debug_text += f" | Loaded: {self.video.loaded}"
                    
                self.time_label.text = debug_text[:50]  # Truncate for display
                
                # Update position slider if video is loaded
                if hasattr(self.video, 'duration') and self.video.duration > 0:
                    progress = (self.video.position / self.video.duration) * 100
                    self.position_slider.value = progress
                    
            except Exception as e:
                self.time_label.text = f"Update Error: {str(e)[:20]}"
    
    def update_status(self):
        """Update status label with current info"""
        try:
            video_path = os.path.abspath('sample2.mp4')
            status_text = f"File: sample2.mp4\n"
            status_text += f"Exists: {os.path.exists(video_path)}\n"
            if hasattr(self.video, 'state'):
                status_text += f"State: {self.video.state}\n"
            if hasattr(self.video, 'loaded'):
                status_text += f"Loaded: {self.video.loaded}"
            
            self.status_label.text = status_text
            self.status_label.text_size = (400, None)
        except Exception as e:
            self.status_label.text = f"Status Error: {e}"
    
    def on_stop(self):
        """Called when app is closing"""
        print("App closing...")
        if self.video and hasattr(self.video, 'state'):
            self.video.state = 'stop'


if __name__ == '__main__':
    # Enhanced debugging
    video_path = os.path.abspath('sample2.mp4')
    print("="*50)
    print("VIDEO DEBUG INFORMATION")
    print("="*50)
    print(f"Current directory: {os.getcwd()}")
    print(f"Video path: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}")
    
    if os.path.exists(video_path):
        print(f"File size: {os.path.getsize(video_path)} bytes")
        print(f"File extension: {os.path.splitext(video_path)[1]}")
    else:
        print("ERROR: Video file not found!")
        print("Available files in current directory:")
        for f in os.listdir('.'):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  - {f}")
    
    print("="*50)
    
    # Check Kivy video backends
    try:
        from kivy.core.video import Video as CoreVideo
        print(f"Available video providers: {CoreVideo.get_providers()}")
    except Exception as e:
        print(f"Error checking video providers: {e}")
    
    print("Starting app...")
    DebugVideoPlayerApp().run()