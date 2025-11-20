# Bullet Impact Detection System

Python application for real-time detection and tracking of bullet impacts on paper targets using computer vision.

## Features

### Basic Detector (`bullet_impact_detector.py`)
- Real-time bullet hole detection using frame differencing
- Contour-based impact identification
- Impact counting and location tracking
- Visual annotation of detected impacts
- Frame capture and logging

### Advanced Detector (`advanced_detector.py`)
- Image stabilization to handle camera shake
- Automatic scoring system with configurable rings
- Score tracking and statistics
- JSON export of shooting results
- Enhanced visualization with color-coded impacts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Detector

```bash
python bullet_impact_detector.py
```

**Controls:**
- `b` - Capture baseline (clean target before shooting)
- `r` - Reset all detections
- `s` - Save current annotated frame
- `q` - Quit application

### Advanced Detector

```bash
python advanced_detector.py
```

**Controls:**
- `b` - Capture baseline
- `e` - Export results to JSON
- `r` - Reset
- `q` - Quit

## How It Works

1. **Baseline Capture**: Capture reference image of clean target
2. **Frame Processing**: Apply preprocessing (grayscale, blur, threshold)
3. **Difference Detection**: Compare current frame with reference
4. **Contour Analysis**: Identify bullet holes by shape and size
5. **Impact Tracking**: Record location, timestamp, and score
6. **Visualization**: Display annotated video with impact markers

## Configuration

### Detection Parameters

```python
detector = BulletImpactDetector(
    video_source=0,          # Camera index or video file
    min_hole_area=50,        # Minimum hole size in pixels
    max_hole_area=2000       # Maximum hole size in pixels
)
```

### Scoring Rings (Advanced)

Modify `scoring_rings` in `advanced_detector.py`:

```python
self.scoring_rings = [
    {'radius': 50, 'points': 10, 'name': 'Bullseye'},
    {'radius': 100, 'points': 9, 'name': 'Ring 9'},
    # Add more rings as needed
]
```

## Tips for Best Results

1. **Lighting**: Ensure consistent, even lighting on the target
2. **Camera Position**: Mount camera perpendicular to target
3. **Baseline**: Capture baseline before any shots are fired
4. **Distance**: Adjust `min_hole_area` and `max_hole_area` based on camera distance
5. **Threshold**: Modify `diff_threshold` if detection is too sensitive/insensitive

## Troubleshooting

**Issue**: False detections
- Increase `diff_threshold` value
- Adjust `min_hole_area` to filter small noise
- Ensure stable camera mounting

**Issue**: Missing detections
- Decrease `diff_threshold` value
- Check lighting conditions
- Verify `min_hole_area` and `max_hole_area` ranges

**Issue**: Duplicate detections
- Detection cooldown is built-in
- Increase `min_distance` parameter in `_is_new_impact()`

## Output Format

### Console Output
```
Shot #1: Position (320, 240), Area: 156px, Circularity: 0.85
Shot #2: 9 points (Ring 9)
```

### JSON Export (Advanced Detector)
```json
{
  "total_shots": 10,
  "total_score": 87,
  "average_score": 8.7,
  "shots": [
    {
      "shot_number": 1,
      "x": 320,
      "y": 240,
      "score": 9,
      "ring": "Ring 9",
      "timestamp": "2025-11-12T14:30:45.123456"
    }
  ]
}
```

## Future Enhancements

- [ ] Deep learning-based detection (YOLO/Faster R-CNN)
- [ ] Multi-target support
- [ ] Group size calculation
- [ ] Heat map visualization
- [ ] Web dashboard for results
- [ ] Mobile app integration
- [ ] Caliber detection
- [ ] Automatic target recognition

## License

MIT License
