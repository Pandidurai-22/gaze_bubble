import cv2
import numpy as np
import pygame
from pygame.locals import *
import time
from scipy.ndimage import gaussian_filter
import math
from gaze_smoothing import GazeSmoothing
import mediapipe as mp
from sklearn.linear_model import LinearRegression
import numpy as np


pygame.init()


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BUBBLE_RADIUS = 30
BUBBLE_COLOR = (0, 150, 255, 128)  # RGBA
BUBBLE_TRAIL_LENGTH = 10
SMOOTHING_ALPHA = 0.7


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Eye Gaze Bubble Animation")

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
        
        pygame.draw.circle(surface, (*BUBBLE_COLOR[:3], self.alpha), (int(self.x), int(self.y)), BUBBLE_RADIUS)
        
        
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(self.trail_colors[i] * (1 - i/BUBBLE_TRAIL_LENGTH))
            color = (*BUBBLE_COLOR[:3], alpha)
            radius = int(BUBBLE_RADIUS * (1 - i/BUBBLE_TRAIL_LENGTH))
            pygame.draw.circle(surface, color, (int(tx), int(ty)), radius)

class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        
        # Initialize GazeSmoothing with compatibility for different versions
        try:
            self.smoother = GazeSmoothing(alpha=SMOOTHING_ALPHA)
        except TypeError:
            # Fallback for versions that don't support alpha parameter
            self.smoother = GazeSmoothing()
            if hasattr(self.smoother, 'alpha'):
                self.smoother.alpha = SMOOTHING_ALPHA
                
        self.screen_center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
        
        
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]
        
        self.head_movement_threshold = 0.05  # 5% of screen width
        self.previous_head_position = None
        
       
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
        
    # def get_gaze_position(self):
    #     """Get gaze position using iris tracking"""
    #     success, image = self.cap.read()
    #     if not success:
    #         return None
            
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     results = self.face_mesh.process(image)
        
    #     if results.multi_face_landmarks:
    #         landmarks = results.multi_face_landmarks[0].landmark
            
    #         # Get head position (using nose tip)
    #         nose_tip = landmarks[1]
    #         current_head_position = (nose_tip.x, nose_tip.y)
            
    #         # Check if head movement is too large
    #         if self.previous_head_position is not None:
    #             head_movement = abs(current_head_position[0] - self.previous_head_position[0])
    #             if head_movement > self.head_movement_threshold:
    #                 return None  # Discard if head movement is too large
            
    #         self.previous_head_position = current_head_position
            
    #         try:
    #             # Get left eye and iris landmarks
    #             left_eye_corners = [landmarks[p] for p in self.LEFT_EYE]
    #             left_iris = np.mean([[landmarks[p].x, landmarks[p].y] for p in self.LEFT_IRIS], axis=0)
                
    #             # Get right eye and iris landmarks
    #             right_eye_corners = [landmarks[p] for p in self.RIGHT_EYE]
    #             right_iris = np.mean([[landmarks[p].x, landmarks[p].y] for p in self.RIGHT_IRIS], axis=0)
                

    #             # Horizontal gaze (X-axis)
    #             left_eye_width = abs(left_eye_corners[1].x - left_eye_corners[0].x)
    #             right_eye_width = abs(right_eye_corners[1].x - right_eye_corners[0].x)

    #             left_ratio_x = (left_iris[0] - left_eye_corners[0].x) / left_eye_width
    #             right_ratio_x = (right_iris[0] - right_eye_corners[0].x) / right_eye_width
    #             gaze_ratio_x = 1 - ((left_ratio_x + right_ratio_x) / 2)  # INVERTED

    #             # Vertical gaze (Y-axis) — use eye height
    #             left_eye_top = landmarks[159]   # approximate top
    #             left_eye_bottom = landmarks[145] # approximate bottom
    #             right_eye_top = landmarks[386]
    #             right_eye_bottom = landmarks[374]

    #             left_eye_height = abs(left_eye_bottom.y - left_eye_top.y)
    #             right_eye_height = abs(right_eye_bottom.y - right_eye_top.y)

    #             left_ratio_y = (left_iris[1] - left_eye_top.y) / left_eye_height
    #             right_ratio_y = (right_iris[1] - right_eye_top.y) / right_eye_height
    #             gaze_ratio_y = 1 - ((left_ratio_y + right_ratio_y) / 2)  # INVERTED

    #             # Map to screen coordinates
    #             screen_x = SCREEN_WIDTH * gaze_ratio_x
    #             screen_y = SCREEN_HEIGHT * gaze_ratio_y


                
    #             # Apply smoothing
    #             screen_x, screen_y = self.smoother.smooth_angles(screen_x, screen_y)
                
    #             return (screen_x, screen_y)
                
    #         except Exception as e:
    #             print(f"Error processing landmarks: {str(e)}")
    #             return None
    #     return None

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
            # Get left eye and iris landmarks
            left_eye_corners = [landmarks[p] for p in self.LEFT_EYE]
            left_iris = np.mean([[landmarks[p].x, landmarks[p].y] for p in self.LEFT_IRIS], axis=0)
            
            # Get right eye and iris landmarks
            right_eye_corners = [landmarks[p] for p in self.RIGHT_EYE]
            right_iris = np.mean([[landmarks[p].x, landmarks[p].y] for p in self.RIGHT_IRIS], axis=0)
            
            # Horizontal gaze (X-axis)
            left_eye_width = abs(left_eye_corners[1].x - left_eye_corners[0].x)
            right_eye_width = abs(right_eye_corners[1].x - right_eye_corners[0].x)
            
            left_ratio_x = (left_iris[0] - left_eye_corners[0].x) / left_eye_width
            right_ratio_x = (right_iris[0] - right_eye_corners[0].x) / right_eye_width
            gaze_ratio_x = 1 - ((left_ratio_x + right_ratio_x) / 2)  # INVERTED
            
            # Vertical gaze (Y-axis)
            left_eye_top = landmarks[159]   # approximate top
            left_eye_bottom = landmarks[145] # approximate bottom
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            
            left_eye_height = abs(left_eye_bottom.y - left_eye_top.y)
            right_eye_height = abs(right_eye_bottom.y - right_eye_top.y)
            
            left_ratio_y = (left_iris[1] - left_eye_top.y) / left_eye_height
            right_ratio_y = (right_iris[1] - right_eye_top.y) / right_eye_height
            gaze_ratio_y = 1 - ((left_ratio_y + right_ratio_y) / 2)  # INVERTED
            
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
        
        
        if calibration_data:
            X = np.array([ratios for ratios, _ in calibration_data])
            y_x = np.array([screen_pos[0] for _, screen_pos in calibration_data])
            y_y = np.array([screen_pos[1] for _, screen_pos in calibration_data])
            
            self.model_x = LinearRegression().fit(X, y_x)
            self.model_y = LinearRegression().fit(X, y_y)
            self.calibrated = True
            return True
            
        return False
    
    def get_gaze_position(self):
        if not self.calibrated:
            return None
            
        gaze_ratio_x, gaze_ratio_y = self.get_raw_gaze_ratios()
        if gaze_ratio_x is None or gaze_ratio_y is None:
            return None
            
        # Use the calibration model to map ratios to screen coordinates
        input_ratios = np.array([[gaze_ratio_x, gaze_ratio_y]])
        screen_x = self.model_x.predict(input_ratios)[0]
        screen_y = self.model_y.predict(input_ratios)[0]
        
        # Apply smoothing
        screen_x, screen_y = self.smoother.smooth_angles(screen_x, screen_y)
        
        # Clamp to screen bounds
        screen_x = max(0, min(SCREEN_WIDTH, screen_x))
        screen_y = max(0, min(SCREEN_HEIGHT, screen_y))
        
        return (screen_x, screen_y)


def show_message(screen, message, y_offset=0, font_size=36, color=(255, 255, 255)):
    font = pygame.font.Font(None, font_size)
    text = font.render(message, True, color)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset))
    screen.blit(text, text_rect)

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
    # Initialize tracker and bubble
    eye_tracker = EyeTracker()
    bubble = Bubble(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
    clock = pygame.time.Clock()
    
    # Load and scale background image
    try:
        background = pygame.image.load('background_new.png')
        background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
    except pygame.error as e:
        print(f"Error loading background image: {e}")
        background = None
    
    # Show instructions and wait for user to start calibration
    if not show_instructions(screen):
        pygame.quit()
        return
    
    # Run calibration
    if not eye_tracker.calibrate(screen):
        print("Calibration failed or was cancelled.")
        pygame.quit()
        return
    
    # Main tracking loop
    current_zone = None
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        
        # Get gaze position
        gaze_pos = eye_tracker.get_gaze_position()
        
        # Update bubble position if gaze detected
        if gaze_pos:
            screen_x, screen_y = gaze_pos
            bubble.update(screen_x, screen_y)
            
            # Determine gaze zone (optional)
            if screen_x < SCREEN_WIDTH / 3:
                new_zone = "Left"
            elif screen_x < 2 * SCREEN_WIDTH / 3:
                new_zone = "Center"
            else:
                new_zone = "Right"
                
            # Print zone only if it has changed
            if new_zone != current_zone:
                print(f"Looking at {new_zone} zone")
                current_zone = new_zone
        
        # Draw everything
        if background:
            screen.blit(background, (0, 0))  # Draw background image
        else:
            screen.fill((0, 0, 0))  # Fallback to black background
            
        bubble.draw(screen)  # Draw bubble on top of background
        
        # Show status
        font = pygame.font.Font(None, 36)
        status_text = "Tracking - Press ESC to exit"
        text = font.render(status_text, True, (255, 255, 255))
        screen.blit(text, (20, 20))
        
        pygame.display.flip()
        clock.tick(60)  # Cap at 60 FPS
    
    # Clean up
    eye_tracker.cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
