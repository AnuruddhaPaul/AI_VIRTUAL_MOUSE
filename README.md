# AI Virtual Mouse

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

AI Virtual Mouse is a computer vision-based system that allows users to control their cursor using hand gestures captured by a webcam. This project leverages **MediaPipe** for hand tracking and **PyAutoGUI** for mouse control.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Gesture Controls](#gesture-controls)
- [Code Explanation](#code-explanation)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

This project implements a virtual mouse controlled by hand gestures. Using your webcam, you can:

- **Move the cursor** by pointing with your index finger.
- **Perform left-clicks** by bringing your index and middle fingers together.
- **Perform right-clicks** by bringing your index finger and thumb together.
- **Exit the program** by pressing the `q` key.

---

## Requirements

Ensure you have the following installed:

- Python 3.6+
- OpenCV
- NumPy
- MediaPipe
- PyAutoGUI
- A working webcam

---

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/ai-virtual-mouse.git
cd ai-virtual-mouse
```

Install the required dependencies:

```bash
pip install opencv-python numpy mediapipe pyautogui
```

---

## Usage

Run the main script:

```bash
python AI_VIRTUAL_MOUSE.py
```

---

## Gesture Controls

| Gesture          | Action                 |
|-----------------|------------------------|
| Index finger up | Move cursor            |
| Index + Middle fingers close | Left Click |
| Index finger + Thumb close   | Right Click |
| Press 'q' key  | Exit program            |

---

## Code Explanation

### `AI_VIRTUAL_MOUSE.py`

This script contains the main logic for hand gesture-based cursor control.

- **Initialization and Setup:**
  ```python
  pyautogui.FAILSAFE = True  # Move mouse to corner to abort
  pyautogui.PAUSE = 0.1  # Small delay between commands
  ```
  - `FAILSAFE`: Move the mouse to the corner to exit safely.
  - `PAUSE`: Prevents excessive execution speed.

- **Helper Functions:**
  - `finger_up(landmarks, finger_idx)`: Checks if a specific finger is raised.
  - `get_fingers_up(landmarks)`: Returns a list of raised fingers.
  - `find_distance(p1, p2, img, draw, r, t)`: Calculates the Euclidean distance between two landmarks.

- **Main Loop Features:**
  - **Hand Detection:**
    ```python
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    ```
  - **Mouse Movement:**
    ```python
    if fingers[1] and not any(fingers[2:]):
        pyautogui.moveTo(clocX, clocY)
    ```
  - **Click Detection:**
    ```python
    if fingers[1] and fingers[2]:
        pyautogui.leftClick()
    if fingers[1] and fingers[0]:
        pyautogui.rightClick()
    ```

### `HAND_TRACKING_MODULE.py`

This module provides a reusable class for detecting and tracking hand movements.

- **Hand Detection:**
  ```python
  def findHands(self, img, draw=True):
      imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.results = self.hands.process(imgRGB)
      return img
  ```
- **Finger Tracking:**
  ```python
  def fingersUp(self):
      return [1 if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2] else 0 for i in range(1, 5)]
  ```

---

## Customization

You can adjust parameters to improve accuracy and responsiveness:

- **Sensitivity:**
  ```python
  frameR = 100  # Adjust frame reduction size
  smoothening = 7  # Adjust smoothening for movement
  ```
- **Click Distances:**
  ```python
  if length < 40:  # Left click threshold
  if length < 80:  # Right click threshold
  ```

---

## Troubleshooting

| Issue                  | Solution                                      |
|------------------------|-----------------------------------------------|
| Poor detection         | Ensure good lighting and a clear background  |
| Erratic cursor movement | Increase smoothening value                  |
| False clicks          | Increase click distance thresholds            |
| No detection         | Check webcam functionality                    |
| Performance issues   | Lower camera resolution or FPS                |

---

## License

This project is licensed under the MIT License.

---

_Last updated: March 17, 2025_
