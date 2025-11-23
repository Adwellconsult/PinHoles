"""
Simple OpenCV Video Player as alternative to Kivy
Uses OpenCV for reliable video playback
"""

import cv2
import threading
import time


class OpenCVVideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.playing = False
        self.paused = False
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        print(f"Video Info:")
        print(f"  Path: {video_path}")
        print(f"  FPS: {self.frame_rate}")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.frame_rate:.2f} seconds")
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
    
    def play(self):
        """Start video playback"""
        if not self.playing:
            self.playing = True
            self.paused = False
            thread = threading.Thread(target=self._play_loop, daemon=True)
            thread.start()
            print("Video started")
    
    def pause(self):
        """Pause/resume video"""
        self.paused = not self.paused
        print(f"Video {'paused' if self.paused else 'resumed'}")
    
    def stop(self):
        """Stop video and reset"""
        self.playing = False
        self.paused = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        print("Video stopped")
    
    def seek(self, frame_number):
        """Seek to specific frame"""
        if 0 <= frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            print(f"Seeked to frame {frame_number}")
    
    def _play_loop(self):
        """Main playback loop"""
        while self.playing:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video
                    self.playing = False
                    print("Video finished")
                    break
                
                self.current_frame += 1
                
                # Display frame
                cv2.imshow('Video Player - sample2.mp4', frame)
                
                # Update window title with progress
                progress = (self.current_frame / self.total_frames) * 100
                current_time = self.current_frame / self.frame_rate
                total_time = self.total_frames / self.frame_rate
                title = f'Video Player - {current_time:.1f}s / {total_time:.1f}s ({progress:.1f}%)'
                cv2.setWindowTitle('Video Player - sample2.mp4', title)
            
            # Handle keyboard input
            key = cv2.waitKey(int(1000 / self.frame_rate)) & 0xFF
            if key == ord('q'):
                self.playing = False
                break
            elif key == ord(' '):  # Space bar
                self.pause()
            elif key == ord('r'):  # Reset/restart
                self.seek(0)
            elif key == ord('s'):  # Stop
                self.stop()
                break
        
        cv2.destroyAllWindows()
    
    def get_info(self):
        """Get current playback information"""
        current_time = self.current_frame / self.frame_rate
        total_time = self.total_frames / self.frame_rate
        progress = (self.current_frame / self.total_frames) * 100
        
        return {
            'current_frame': self.current_frame,
            'total_frames': self.total_frames,
            'current_time': current_time,
            'total_time': total_time,
            'progress': progress,
            'playing': self.playing,
            'paused': self.paused
        }


def main():
    """Main function to run the video player"""
    import os
    
    video_path = 'sample2.mp4'
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Available video files:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                print(f"  - {f}")
        return
    
    try:
        # Create video player
        player = OpenCVVideoPlayer(video_path)
        
        print("\n" + "="*50)
        print("OPENCV VIDEO PLAYER CONTROLS")
        print("="*50)
        print("SPACE - Pause/Resume")
        print("R     - Restart video")
        print("S     - Stop video")
        print("Q     - Quit")
        print("="*50)
        
        # Start playback
        player.play()
        
        # Keep main thread alive and show periodic info
        while player.playing:
            time.sleep(2)
            info = player.get_info()
            if info['playing']:
                print(f"Playing: {info['current_time']:.1f}s / {info['total_time']:.1f}s "
                      f"({info['progress']:.1f}%) - Frame {info['current_frame']}/{info['total_frames']}")
        
        print("Video player finished")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OpenCV is installed: pip install opencv-python")


if __name__ == '__main__':
    main()