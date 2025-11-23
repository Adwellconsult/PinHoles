"""
Simple Kivy Video Player for sample2.mp4
Displays video with basic playback controls
"""

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.video import Video
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.clock import Clock
import os

kivy.require('2.0.0')


class VideoPlayerApp(App):
    def __init__(self, **kwargs):
        super(VideoPlayerApp, self).__init__(**kwargs)
        self.video = None
        self.is_playing = False
        
    def build(self):
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Video widget
        video_path = os.path.join(os.getcwd(), 'sample2.mp4')
        self.video = Video(source=video_path, state='stop', options={'allow_stretch': True})
        
        # Control layout
        control_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        
        # Play/Pause button
        self.play_button = Button(text='Play', size_hint_x=None, width=100)
        self.play_button.bind(on_press=self.toggle_play)
        
        # Stop button
        stop_button = Button(text='Stop', size_hint_x=None, width=100)
        stop_button.bind(on_press=self.stop_video)
        
        # Position slider
        self.position_slider = Slider(min=0, max=100, value=0, step=1)
        self.position_slider.bind(on_touch_up=self.seek_video)
        
        # Time label
        self.time_label = Label(text='00:00 / 00:00', size_hint_x=None, width=120)
        
        # Add controls to layout
        control_layout.add_widget(self.play_button)
        control_layout.add_widget(stop_button)
        control_layout.add_widget(self.position_slider)
        control_layout.add_widget(self.time_label)
        
        # Status label
        self.status_label = Label(text=f'Video: {video_path}', size_hint_y=None, height=30)
        
        # Add all to main layout
        main_layout.add_widget(self.video)
        main_layout.add_widget(control_layout)
        main_layout.add_widget(self.status_label)
        
        # Schedule position update
        Clock.schedule_interval(self.update_position, 0.1)
        
        return main_layout
    
    def toggle_play(self, instance):
        """Toggle between play and pause"""
        if self.video.state == 'play':
            self.video.state = 'pause'
            self.play_button.text = 'Play'
            self.is_playing = False
            self.status_label.text = 'Video Paused'
        else:
            self.video.state = 'play'
            self.play_button.text = 'Pause'
            self.is_playing = True
            self.status_label.text = 'Video Playing'
    
    def stop_video(self, instance):
        """Stop video playback"""
        self.video.state = 'stop'
        self.play_button.text = 'Play'
        self.is_playing = False
        self.position_slider.value = 0
        self.status_label.text = 'Video Stopped'
    
    def seek_video(self, instance, touch):
        """Seek to position based on slider"""
        if instance.collide_point(*touch.pos):
            if self.video.duration > 0:
                seek_pos = (instance.value / 100.0) * self.video.duration
                self.video.seek(seek_pos)
                self.status_label.text = f'Seeking to {seek_pos:.1f}s'
    
    def update_position(self, dt):
        """Update position slider and time display"""
        if self.video.duration > 0:
            # Update slider position
            progress = (self.video.position / self.video.duration) * 100
            self.position_slider.value = progress
            
            # Update time display
            current_time = self.format_time(self.video.position)
            total_time = self.format_time(self.video.duration)
            self.time_label.text = f'{current_time} / {total_time}'
            
            # Check if video ended
            if self.video.position >= self.video.duration and self.is_playing:
                self.stop_video(None)
    
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f'{minutes:02d}:{seconds:02d}'
    
    def on_stop(self):
        """Called when app is closing"""
        if self.video:
            self.video.state = 'stop'


if __name__ == '__main__':
    # Check if video file exists
    video_path = os.path.join(os.getcwd(), 'sample2.mp4')
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please make sure sample2.mp4 is in the current directory")
    else:
        print(f"Loading video: {video_path}")
        VideoPlayerApp().run()