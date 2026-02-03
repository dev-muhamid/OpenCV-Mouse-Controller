import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
from gpu_check import gpu_available

# Set environment variables for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Best-effort GPU availability check (fast; no TensorFlow import by default)
has_gpu = gpu_available(use_tensorflow=False)
print(f"GPU available: {has_gpu}")

# Initialize MediaPipe Hands with GPU
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands with a complexity that matches availability
model_complexity = 1 if has_gpu else 0
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=model_complexity
)
if has_gpu:
    print("MediaPipe initialized - GPU detected (attempting GPU acceleration)")
else:
    print("MediaPipe initialized - no GPU detected (using CPU settings)")

# Function to label hand movements and perform mouse operations
def label_and_control_mouse(frame): 
    global prev_x, prev_y, buffer, first_detection
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert landmarks to screen coordinates
            screen_width, screen_height = pyautogui.size()
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            
            # Map index finger to screen coordinates
            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)

            # Calculate the distance between thumb and index finger
            distance = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            cv2.putText(frame,f"Distance: {distance:.2f}",(10,60), cv2.FONT_HERSHEY_COMPLEX,1,(100,200,100),2)
            cv2.line(frame,(thumb_x, thumb_y),(index_x, index_y),(85, 255, 153 ))

            # Draw circle at tip of index finger
            cv2.circle(frame, (index_x, index_y), 10,(255, 0,0), -1)
            
            # Control the mouse cursor
            if distance < 30:
                pyautogui.click()
            elif distance < 70:
                pyautogui.doubleClick()
            else:
                # Move mouse to index finger position with smoothing
                if first_detection:
                    pyautogui.moveTo(screen_x, screen_y)
                    first_detection = False
                else:
                    buffer.append((screen_x, screen_y))
                    if len(buffer) > 5:
                        buffer.pop(0)
                    avg_x = sum(x for x, y in buffer) / len(buffer)
                    avg_y = sum(y for x, y in buffer) / len(buffer)
                    pyautogui.moveTo(int(avg_x), int(avg_y))

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Initialize    
first_detection = True
buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1680,1050))
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    # Label hand movements and control mouse in the frame
    labeled_frame = label_and_control_mouse(frame)

    # Display the labeled frame
    cv2.imshow('Virtual Mouse', labeled_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()