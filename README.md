# Gaze Bubble

A real-time eye-tracking application that displays a visual bubble overlay on your screen, showing exactly where you're looking. Built with MediaPipe for face detection, OpenCV for camera processing, and Pygame for the transparent overlay display.

## Demo

Watch the demo video to see Gaze Bubble in action:

<img src="assets/GazeBubble_giffile.gif">

</img>

*Note: If you want the full demo video, you can [download it directly](assets/GazeBubble_demo.mp4).*

## Features

- **Real-time Eye Tracking**: Uses MediaPipe Face Mesh to detect and track eye movements
- **Visual Feedback**: Displays a semi-transparent bubble overlay that follows your gaze
- **Calibration System**: 9-point calibration routine using linear regression for accurate gaze mapping
- **Gaze Smoothing**: Exponential moving average smoothing for stable, jitter-free tracking
- **Trail Effect**: Visual trail showing the bubble's movement path
- **Zone Detection**: Automatically detects which screen zone (Left/Center/Right) you're looking at
- **Transparent Overlay**: Always-on-top transparent window that doesn't interfere with your work
- **Windows Integration**: Native Windows API integration for transparency and window management

## Requirements

### System Requirements
- **Operating System**: Windows 10/11 (uses Windows-specific APIs for transparency)
- **Webcam**: A working webcam/camera for eye tracking
- **Python**: Python 3.8 or higher

### Python Dependencies
All dependencies are listed in `requirements.txt`:
- `opencv-python>=4.8.0` - Camera capture and image processing
- `numpy>=1.24.0` - Numerical computations
- `mediapipe==0.10.9` - Face and eye landmark detection
- `pygame>=2.5.0` - Overlay window rendering
- `pillow>=10.0.0` - Image processing support
- `scipy>=1.11.0` - Scientific computing
- `scikit-learn` - Linear regression for calibration
- `pyautogui>=0.9.54` - Screen utilities
- `eyegestures>=1.0.0` - Eye gesture recognition

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd gaze_bubble
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**
   ```bash
   python gaze_bubble/gaze_bubble.py
   ```

2. **Calibration Process**:
   - The application will start with an instruction screen
   - Press **SPACE** to begin calibration
   - Look at each of the 9 calibration points as they highlight (green)
   - Each point will be displayed for 3 seconds
   - Press **ESC** at any time to cancel

3. **Using the Overlay**:
   - After calibration, a transparent overlay window will appear
   - The bubble will follow your gaze in real-time
   - The overlay stays on top of all windows
   - Press **ESC** to exit the application

## Configuration

You can modify the following constants in `gaze_bubble.py` to customize the behavior:

```python
SCREEN_WIDTH = 1920          # Screen width (adjust to your resolution)
SCREEN_HEIGHT = 1080         # Screen height (adjust to your resolution)
BUBBLE_RADIUS = 30           # Size of the gaze bubble
BUBBLE_COLOR = (100, 160, 120, 190)  # RGB color of the bubble
BUBBLE_TRAIL_LENGTH = 10     # Number of trail points to display
SMOOTHING_ALPHA = 0.7        # Smoothing factor (0.0-1.0, higher = less smoothing)
```

### Click-Through Mode

To enable click-through mode (allows mouse clicks to pass through the overlay), uncomment line 341 in `gaze_bubble.py`:

```python
make_click_through(hwnd, click_through=True)
```

## How It Works

### Eye Tracking Pipeline

1. **Face Detection**: MediaPipe Face Mesh detects facial landmarks including eye corners and iris positions
2. **Gaze Ratio Calculation**: Calculates the relative position of the iris within the eye boundaries
3. **Calibration**: Maps gaze ratios to screen coordinates using linear regression on 9 calibration points
4. **Smoothing**: Applies exponential moving average to reduce jitter and noise
5. **Visualization**: Renders a bubble overlay at the predicted gaze position

### Calibration System

The calibration uses a 9-point grid pattern:
- 3 points across the top
- 3 points across the middle
- 3 points across the bottom

For each point, the system:
- Collects gaze ratio data for 3 seconds
- Averages the collected data
- Trains separate linear regression models for X and Y coordinates
- Uses these models to map future gaze ratios to screen positions

### Gaze Smoothing

The `GazeSmoothing` class uses exponential moving average (EMA) to smooth gaze predictions:
- Reduces jitter from camera noise and detection variations
- Configurable smoothing factor (alpha)
- Maintains responsiveness while filtering out rapid fluctuations

## Troubleshooting

### Camera Not Detected
- Ensure your webcam is connected and not being used by another application
- Check that the camera index is correct (default is 0)
- Try running with administrator privileges

### Calibration Issues
- Ensure good lighting conditions
- Keep your face centered and at a consistent distance from the camera
- Make sure your entire face is visible in the camera frame
- Avoid rapid head movements during calibration

### Overlay Not Transparent
- This feature requires Windows 10/11
- Ensure you're running the latest version of Pygame
- Try running with administrator privileges

### Poor Tracking Accuracy
- Re-run calibration if tracking seems off
- Adjust lighting to improve face detection
- Ensure you're sitting at a consistent distance from the camera
- Try adjusting `SMOOTHING_ALPHA` for better stability

### Performance Issues
- Close other applications using the camera
- Reduce `BUBBLE_TRAIL_LENGTH` for better performance
- Lower camera resolution if needed (modify OpenCV capture settings)

## Technical Details

### Windows API Integration

The application uses Windows API calls via `ctypes` to:
- Create layered windows with transparency
- Set window to always-on-top
- Enable click-through functionality (optional)
- Use color keying for transparency (black pixels become transparent)

### MediaPipe Landmarks

The application uses specific MediaPipe face mesh landmarks:
- **Left Eye**: Points 33, 133 (corners)
- **Right Eye**: Points 362, 263 (corners)
- **Left Iris**: Points 469, 470, 471, 472
- **Right Iris**: Points 474, 475, 476, 477

## Limitations

- **Windows Only**: Uses Windows-specific APIs for transparency
- **Single Monitor**: Currently configured for single monitor setups
- **Single User**: Designed for one user at a time
- **Lighting Dependent**: Requires adequate lighting for accurate face detection
- **Camera Quality**: Tracking accuracy depends on camera quality and resolution

## Future Improvements

Potential enhancements:
- Multi-monitor support
- Save/load calibration profiles
- Customizable bubble appearance
- Additional gesture recognition
- Cross-platform support (Linux/macOS)
- Performance optimizations

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


## Author
- Pandi Durai S 
- Software Engineer (1.5+ years Experience)
- pandidurai32127@gmail.com
- +91 9751391299

## Acknowledgments

- MediaPipe for face mesh detection
- Pygame for window management
- OpenCV for camera handling
