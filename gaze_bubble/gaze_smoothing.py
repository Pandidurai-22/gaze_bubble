import numpy as np
from typing import Tuple

class GazeSmoothing:
    def __init__(self, alpha=0.8, **kwargs):
        """
        Initialize gaze smoothing with exponential moving average
        
        Args:
            alpha: Smoothing factor (0.0 to 1.0), higher values give more weight to current prediction
            **kwargs: Additional arguments for compatibility
        """
        # Support both alpha as positional or keyword argument
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = alpha
        self.previous_pitch = None
        self.previous_yaw = None
        
        # Ensure alpha is within valid range
        self.alpha = max(0.0, min(1.0, float(self.alpha)))
        
    def smooth_angles(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """
        Smooth the gaze angles using exponential moving average
        
        Args:
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            
        Returns:
            Tuple of smoothed (pitch, yaw) angles
        """
        if self.previous_pitch is None:
            self.previous_pitch = pitch
        if self.previous_yaw is None:
            self.previous_yaw = yaw
            
        smoothed_pitch = self.alpha * pitch + (1 - self.alpha) * self.previous_pitch
        smoothed_yaw = self.alpha * yaw + (1 - self.alpha) * self.previous_yaw
        
        self.previous_pitch = smoothed_pitch
        self.previous_yaw = smoothed_yaw
        return smoothed_pitch, smoothed_yaw
        
    def reset(self):
        """Reset the smoothing state"""
        self.previous_pitch = None
        self.previous_yaw = None

def smooth_gaze_sequence(gaze_sequence: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """
    Smooth a sequence of gaze predictions
    
    Args:
        gaze_sequence: Array of shape (N, 2) where N is number of frames
        alpha: Smoothing factor
        
    Returns:
        Array of smoothed gaze predictions
    """
    smoother = GazeSmoothing(alpha)
    smoothed_gaze = []
    
    for pitch, yaw in gaze_sequence:
        smoothed = smoother.smooth_angles(pitch, yaw)
        smoothed_gaze.append(smoothed)
    
    return np.array(smoothed_gaze)
