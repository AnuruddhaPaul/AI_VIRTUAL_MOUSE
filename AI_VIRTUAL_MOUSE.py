import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui

# Initialize PyAutoGUI safely
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Add small delay between PyAutoGUI commands

# Camera settings
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7

# Initialize time and location variables
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

def finger_up(landmarks, finger_idx):
    """Check if a finger is up based on landmark positions"""
    if finger_idx == 1:  # Thumb
        return landmarks[finger_idx * 4].x < landmarks[finger_idx * 4 - 1].x
    else:
        return landmarks[finger_idx * 4].y < landmarks[finger_idx * 4 - 2].y

def get_fingers_up(landmarks):
    """Return list of which fingers are up"""
    return [
        finger_up(landmarks, 1),  # Thumb
        finger_up(landmarks, 2),  # Index
        finger_up(landmarks, 3),  # Middle
        finger_up(landmarks, 4),  # Ring
        finger_up(landmarks, 5),  # Pinky
    ]

def find_distance(p1, p2, img=None, draw=True, r=15, t=3):
    """Find distance between two landmarks"""
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    if draw and img is not None:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    
    length = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    return length, [x1, y1, x2, y2, cx, cy] if img is not None else None

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera")
        break
        
    # Flip image for more intuitive movement
    img = cv2.flip(img, 1)
    
    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process hand detection
    results = hands.process(imgRGB)
    
    # Draw boundary box
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    # If hands detected
    if results.multi_hand_landmarks:
        # For the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Get landmark positions
        landmarks = []
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
        
        # Get index and middle finger tips
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]
        
        # Check which fingers are up
        fingers = get_fingers_up(hand_landmarks.landmark)
        print(fingers)
        
        # Moving Mode - Index finger up, middle finger down
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and not fingers[0]:
            # Convert coordinates from cam to screen
            x3 = np.interp(index_tip[0], (frameR, wCam - frameR), (0, screen_width))
            y3 = np.interp(index_tip[1], (frameR, hCam - frameR), (0, screen_height))
            
            # Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            
            # Move mouse
            try:
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY
                cv2.circle(img, index_tip, 15, (255, 0, 255), cv2.FILLED)
            except Exception as e:
                print(f"Error moving mouse: {e}")
        
        # Clicking Mode - Both index and middle fingers up
        
        if fingers[1] and fingers[2]:
            # Find distance between fingers
            length, line_info = find_distance(index_tip, middle_tip, img)
            
            # Click mouse if distance short
            if length < 40:
                print("Left Click")
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.leftClick()
                # Add small delay to avoid multiple clicks
                time.sleep(0.3)
        if fingers[1] and fingers[0]:
            # Find distance between fingers
            length, line_info = find_distance(index_tip, thumb_tip, img)
            
            # Click mouse if distance short
            if length < 80:
                print("Right Click")
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.rightClick()
                # Add small delay to avoid multiple clicks
                time.sleep(0.3)
    
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Display the imageq
    cv2.imshow("Virtual Mouse", img)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

