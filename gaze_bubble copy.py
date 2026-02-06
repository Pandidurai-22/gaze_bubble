import cv2
import numpy as np
import pygame
from pygame.locals import *
import time
from gaze_smoothing import GazeSmoothing
import mediapipe as mp
from sklearn.linear_model import LinearRegression
import ctypes
from ctypes import wintypes


HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
GWL_EXSTYLE = -20
LWA_ALPHA = 0x2


user32 = ctypes.windll.user32
SetWindowLong = user32.SetWindowLongW
GetWindowLong = user32.GetWindowLongW
SetLayeredWindowAttributes = user32.SetLayeredWindowAttributes
SetWindowPos = user32.SetWindowPos
FindWindow = user32.FindWindowW

pygame.init()

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BUBBLE_RADIUS = 30
BUBBLE_COLOR = (100, 160, 120, 190)
BUBBLE_TRAIL_LENGTH = 10
SMOOTHING_ALPHA = 0.7
OVERLAY_ALPHA = 0

def setup_transparent_overlay(screen):
   
    hwnd = pygame.display.get_wm_info()['window']
    
  
    style = GetWindowLong(hwnd, GWL_EXSTYLE)
    
    # Add layered style (required for transparency)
    style = style | WS_EX_LAYERED
    SetWindowLong(hwnd, GWL_EXSTYLE, style)
    

    LWA_COLORKEY = 0x1
    # Convert RGB(0,0,0) to COLORREF format: 0x00BBGGRR
    color_key = 0x00000000  # Black in COLORREF format
    SetLayeredWindowAttributes(hwnd, color_key, 0, LWA_COLORKEY)
    
   
    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, 
                 SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
    
    return hwnd

    

def make_click_through(hwnd, click_through=True):
 
    style = GetWindowLong(hwnd, GWL_EXSTYLE)
    if click_through:
        style = style | WS_EX_TRANSPARENT
    else:
        style = style & ~WS_EX_TRANSPARENT
    SetWindowLong(hwnd, GWL_EXSTYLE, style)

class Bubble:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.trail = []
        self.trail_colors = []
        self.alpha = 255
        
    def update(self, new_x, new_y):
        self.trail.append((self.x, self.y))
        self.trail_colors.append(self.alpha)
        if len(self.trail) > BUBBLE_TRAIL_LENGTH:
            self.trail.pop(0)
            self.trail_colors.pop(0)
        self.x = new_x
        self.y = new_y
        
    def draw(self, surface):
        # Draw trail first (so main bubble appears on top)
        trail_len = len(self.trail)
        if trail_len > 0 and BUBBLE_TRAIL_LENGTH > 0:
            for i, (tx, ty) in enumerate(self.trail):
                alpha = int(self.trail_colors[i] * (1 - i / BUBBLE_TRAIL_LENGTH))
                # Use RGB color (no alpha channel needed with color key)
                color = BUBBLE_COLOR[:3]
                radius = int(BUBBLE_RADIUS * (1 - i / BUBBLE_TRAIL_LENGTH))
                pygame.draw.circle(surface, color, (int(tx), int(ty)), radius)
        
        # Draw main bubble (use RGB color, not RGBA)
        pygame.draw.circle(surface, BUBBLE_COLOR[:3], (int(self.x), int(self.y)), BUBBLE_RADIUS)



class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        
        try:
            self.smoother = GazeSmoothing(alpha=SMOOTHING_ALPHA)
        except TypeError:
            self.smoother = GazeSmoothing()
            if hasattr(self.smoother, 'alpha'):
                self.smoother.alpha = SMOOTHING_ALPHA
                
        self.screen_center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
        
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]
        
        self.model_x = None
        self.model_y = None
        self.calibrated = False
        
        self.calibration_points = [
            (100, 100), (SCREEN_WIDTH // 2, 100), (SCREEN_WIDTH - 100, 100),
            (100, SCREEN_HEIGHT // 2), (self.screen_center[0], self.screen_center[1]), 
            (SCREEN_WIDTH - 100, SCREEN_HEIGHT // 2),
            (100, SCREEN_HEIGHT - 100), (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100), 
            (SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)
        ]
        
    def get_raw_gaze_ratios(self):
        """Get raw gaze ratios without any calibration"""
        success, image = self.cap.read()
        if not success:
            return None, None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            return None, None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            left_eye_corners = [landmarks[p] for p in self.LEFT_EYE]
            left_iris = np.mean([[landmarks[p].x, landmarks[p].y] for p in self.LEFT_IRIS], axis=0)
            
            right_eye_corners = [landmarks[p] for p in self.RIGHT_EYE]
            right_iris = np.mean([[landmarks[p].x, landmarks[p].y] for p in self.RIGHT_IRIS], axis=0)
            
            left_eye_width = abs(left_eye_corners[1].x - left_eye_corners[0].x)
            right_eye_width = abs(right_eye_corners[1].x - right_eye_corners[0].x)
            
            if left_eye_width == 0 or right_eye_width == 0:
                return None, None
                
            left_ratio_x = (left_iris[0] - left_eye_corners[0].x) / left_eye_width
            right_ratio_x = (right_iris[0] - right_eye_corners[0].x) / right_eye_width
            gaze_ratio_x = 1 - ((left_ratio_x + right_ratio_x) / 2)
            
            left_eye_top = landmarks[159]   
            left_eye_bottom = landmarks[145] 
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            
            left_eye_height = abs(left_eye_bottom.y - left_eye_top.y)
            right_eye_height = abs(right_eye_bottom.y - right_eye_top.y)
            
            if left_eye_height == 0 or right_eye_height == 0:
                return None, None
                
            left_ratio_y = (left_iris[1] - left_eye_top.y) / left_eye_height
            right_ratio_y = (right_iris[1] - right_eye_top.y) / right_eye_height
            gaze_ratio_y = 1 - ((left_ratio_y + right_ratio_y) / 2)
            
            return gaze_ratio_x, gaze_ratio_y
            
        except Exception as e:
            print(f"Error getting raw gaze ratios: {str(e)}")
            return None, None
    
    def calibrate(self, screen):
        """Run calibration routine"""
        calibration_data = []
        font = pygame.font.Font(None, 36)
        
        for idx, (x, y) in enumerate(self.calibration_points):
            point_data = []
            start_time = time.time()
            
            while time.time() - start_time < 3:  
                gaze_x, gaze_y = self.get_raw_gaze_ratios()
                if gaze_x is not None and gaze_y is not None:
                    point_data.append((gaze_x, gaze_y))
                
                screen.fill((0, 0, 0))
                
                for px, py in self.calibration_points:
                    color = (0, 255, 0) if (px == x and py == y) else (100, 100, 100)
                    pygame.draw.circle(screen, color, (px, py), 10)
                
                text = font.render(f"Look at point {idx+1}/{len(self.calibration_points)}", True, (255, 255, 255))
                screen.blit(text, (SCREEN_WIDTH // 2 - 100, 30))
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        return False
            
            if point_data:
                avg_x = sum([p[0] for p in point_data]) / len(point_data)
                avg_y = sum([p[1] for p in point_data]) / len(point_data)
                calibration_data.append(([avg_x, avg_y], [x, y]))
        
        if len(calibration_data) < 3:
            print("Error: Not enough calibration data collected")
            return False
            
        try:
            X = np.array([ratios for ratios, _ in calibration_data])
            y_x = np.array([screen_pos[0] for _, screen_pos in calibration_data])
            y_y = np.array([screen_pos[1] for _, screen_pos in calibration_data])
            
            self.model_x = LinearRegression().fit(X, y_x)
            self.model_y = LinearRegression().fit(X, y_y)
            self.calibrated = True
            return True
        except Exception as e:
            print(f"Error training calibration models: {e}")
            return False
    
    def get_gaze_position(self):
        if not self.calibrated:
            return None
            
        gaze_ratio_x, gaze_ratio_y = self.get_raw_gaze_ratios()
        if gaze_ratio_x is None or gaze_ratio_y is None:
            return None
        
        try:
            input_ratios = np.array([[gaze_ratio_x, gaze_ratio_y]])
            screen_x = self.model_x.predict(input_ratios)[0]
            screen_y = self.model_y.predict(input_ratios)[0]
        except Exception as e:
            print(f"Error predicting gaze position: {e}")
            return None
        
        screen_x, screen_y = self.smoother.smooth_angles(screen_x, screen_y)
        
        screen_x = max(0, min(SCREEN_WIDTH, screen_x))
        screen_y = max(0, min(SCREEN_HEIGHT, screen_y))
        
        return (screen_x, screen_y)
    
    def release(self):
        if self.cap is not None:
            self.cap.release()

def show_instructions(screen):
    font = pygame.font.Font(None, 36)
    instructions = [
        "Eye Gaze Calibration",
        "",
        "The calibration will show 9 points on the screen.",
        "Please look at each point when it highlights.",
        "",
        "Press SPACE to begin calibration",
        "Press ESC to exit"
    ]
    
    screen.fill((0, 0, 0))
    for i, line in enumerate(instructions):
        text = font.render(line, True, (255, 255, 255))
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 200 + i * 40))
        screen.blit(text, text_rect)
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False
            elif event.type == KEYDOWN and event.key == K_SPACE:
                return True

def main():
    pygame.init()
    
    # Create calibration window (normal window)
    calibration_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Eye Gaze Calibration")
    
    eye_tracker = None
    try:
        eye_tracker = EyeTracker()
        
        # Show instructions
        if not show_instructions(calibration_screen):
            pygame.quit()
            return
        
        # Run calibration
        if not eye_tracker.calibrate(calibration_screen):
            print("Calibration failed or was cancelled.")
            pygame.quit()
            return
        
        # Close calibration window and create overlay
        pygame.quit()
        pygame.init()
        
        # Create transparent overlay window
        # Note: FULLSCREEN might not work well with transparency, using borderless window instead
        overlay_screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), 
            pygame.NOFRAME | pygame.FULLSCREEN
        )
        pygame.display.set_caption("Gaze Overlay")
        
        # Setup transparency and always-on-top
        # Need to flip once to create the window before setting transparency
        overlay_screen.fill((0, 0, 0))
        pygame.display.flip()
        
        hwnd = setup_transparent_overlay(overlay_screen)
        
        # Optional: Make click-through (uncomment to enable)
        # make_click_through(hwnd, click_through=True)
        
        bubble = Bubble(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        clock = pygame.time.Clock()
        
        current_zone = None
        running = True
        
        print("Overlay active! Press ESC to exit.")
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
            
            gaze_pos = eye_tracker.get_gaze_position()
            
            if gaze_pos:
                screen_x, screen_y = gaze_pos
                bubble.update(screen_x, screen_y)
                
                if screen_x < SCREEN_WIDTH / 3:
                    new_zone = "Left"
                elif screen_x < 2 * SCREEN_WIDTH / 3:
                    new_zone = "Center"
                else:
                    new_zone = "Right"
                    
                if new_zone != current_zone:
                    print(f"Looking at {new_zone} zone")
                    current_zone = new_zone
            
            # Clear screen with black (which will be made transparent by color key)
            overlay_screen.fill((0, 0, 0))  # Black = transparent via color key
            
            # Draw only the bubble
            bubble.draw(overlay_screen)
            
            pygame.display.flip()
            clock.tick(60)
            
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if eye_tracker:
            eye_tracker.release()
        pygame.quit()

if __name__ == "__main__":
    main()