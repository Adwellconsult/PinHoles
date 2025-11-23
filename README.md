# Bullet Impact Detector for Android

A professional mobile application for bullet impact detection and scoring using computer vision and target analysis.

## 🎯 Features

- **Precise Target Setup**: Red crosshair cursor for accurate center selection
- **Multiple Scoring Rings**: Define custom scoring zones with visual feedback
- **Video & Camera Support**: Works with live camera or video files
- **Real-time Detection**: Instant bullet impact detection and scoring
- **Professional UI**: Mobile-optimized interface with aspect ratio preservation
- **Smart Video Control**: Pauses on first frame for accurate baseline setup
- **Export Results**: Save detection results and scoring data

## 📱 Installation

### Option 1: Download Pre-built APK
1. Go to the [Releases](../../releases) page
2. Download the latest APK file
3. Enable "Install from unknown sources" on your Android device
4. Install the APK and grant camera permissions

### Option 2: Install from Releases Branch
1. Visit the [`releases` branch](../../tree/releases)
2. Download `BulletDetector-latest.apk`
3. Install on your Android device

### Option 3: Build from Source
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PinHoles.git
cd PinHoles

# Install dependencies
pip install -r requirements.txt
pip install buildozer

# Build APK
buildozer android debug
```

## 🔧 System Requirements

- **Android**: 5.0+ (API 21+)
- **Architecture**: ARM64/ARMv7
- **Permissions**: Camera access
- **Storage**: ~100MB free space

## 🚀 Quick Start

1. **Launch App** → Main menu appears
2. **Settings** → Configure video source (camera/file)
3. **Baseline Setup** → Define target center and scoring rings
4. **Detection** → Start monitoring for bullet impacts
5. **Results** → View scores and export data

## 📖 Usage Guide

### Target Setup
1. Touch target center (red crosshair guides placement)
2. Touch ring edges to define scoring zones
3. Rings are visible during setup, hidden during operation
4. Press "Complete Setup" when finished

### Detection Mode
- Real-time impact detection
- Automatic scoring based on defined rings
- Visual feedback for detected impacts
- Running score display

## 🛠️ Development

### Local Development
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run desktop version
python bullet_detector_android.py
```

### Building for Android
```bash
# Install buildozer
pip install buildozer

# Initialize buildozer (first time only)
buildozer init

# Build debug APK
buildozer android debug

# Build release APK (requires signing)
buildozer android release
```

## 🔄 CI/CD Pipeline

The project includes automated building via GitHub Actions:

- **Triggers**: Push to main/develop, tags, manual dispatch
- **Builds**: Android APK with Buildozer
- **Deploys**: To releases branch and GitHub Releases
- **Artifacts**: APK files with build metadata

### Workflow Features
- Automated APK generation
- Version management from buildozer.spec
- Caching for faster builds
- Release creation for tagged versions
- Deployment to releases branch

## 📁 Project Structure

```
PinHoles/
├── bullet_detector_android.py    # Main Android app
├── buildozer.spec               # Android build configuration
├── requirements.txt             # Python dependencies
├── .github/
│   └── workflows/
│       └── build-android.yml    # CI/CD pipeline
├── stills/
│   └── sample2/                # Sample video files
└── README.md                   # This file
```

## Legacy Desktop Applications

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
