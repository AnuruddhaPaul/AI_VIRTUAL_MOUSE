# AI_VIRTUAL_MOUSE
Hand Tracking Projects

A collection of computer vision applications using hand tracking technology. This repository includes a reusable hand tracking module and applications built on top of it.

Table of Contents

Overview

Requirements

Installation

Modules

HAND_TRACKING_MODULE.py

VIRTUAL_PAINTER.py

Usage

Customization

Troubleshooting

Overview

This project leverages MediaPipe's hand tracking capabilities to create interactive applications controlled by hand gestures. The repository currently includes:

A hand tracking module for detecting and analyzing hand landmarks

A virtual painter application that lets you draw on screen using finger movements

Requirements

Python 3.6+

OpenCV

NumPy

MediaPipe

Webcam

Installation

Clone this repository:

git clone https://github.com/yourusername/hand-tracking-projects.git
cd hand-tracking-projects

Install the required dependencies:

pip install opencv-python numpy mediapipe

Modules

HAND_TRACKING_MODULE.py

A utility module that provides hand detection and tracking functionality.

Class: handDetector

def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

Initializes MediaPipe hand detection with configurable parameters

mode: Processing mode (static image or video)

maxHands: Maximum number of hands to detect

detectionCon: Minimum detection confidence threshold

trackCon: Minimum tracking confidence threshold

findHands(img, draw=True)

Converts image to RGB format

Processes image to detect hands

Draws landmarks and connections

Returns processed image

findPosition(img, handNo=0, draw=True)

Extracts landmark positions

Creates a bounding box around the hand

Draws landmarks and bounding box

Returns landmark list and bounding box

fingersUp()

Determines which fingers are raised

Returns a list of 0s and 1s (0 = down, 1 = up) for each finger

findDistance(p1, p2, img, draw=True, r=15, t=3)

Calculates Euclidean distance between two landmarks

Draws visualization lines and circles

Returns distance, processed image, and point information

VIRTUAL_PAINTER.py

An application that uses the hand tracking module to create a virtual painting experience.

Key Components

Initialization:

brushThickness = 15
eraserThickness = 100
drawColor = (255, 0, 255)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

Sets brush and eraser thickness

Loads header images with color selection options

Initializes drawing canvas as a black image

Selection Mode: (Raise both index and middle fingers)

if fingers[1] and fingers[2]:
    print("Selection Mode")

Detects when fingers are in the header area

Changes drawing color based on selected region

Drawing Mode: (Raise only index finger)

if fingers[1] and fingers[2] == False:
    print("Drawing Mode")

Draws lines on the canvas based on finger movement

Uses a thicker line for eraser when black color is selected

Usage

Hand Tracking Module

To use the hand tracking module in your own projects:

import HAND_TRACKING_MODULE as htm

detector = htm.handDetector()
img = detector.findHands(img)
lmList, bbox = detector.findPosition(img)
fingers = detector.fingersUp()
length, img, lineInfo = detector.findDistance(4, 8, img)

Virtual Painter

To run the virtual painter:

python VIRTUAL_PAINTER.py

Selection Mode: Raise both index and middle fingers to select a color

Drawing Mode: Raise only index finger to draw

Eraser Mode: Select black color and use drawing mode

Exit: Press 'q' on the keyboard

Customization

Brush Size: Modify brushThickness (default: 15)

Eraser Size: Modify eraserThickness (default: 100)

Colors: Add more color options in the header images and update selection logic

Canvas Size: Change dimensions in imgCanvas = np.zeros((720, 1280, 3), np.uint8)

Header Layout: Modify selection mode conditions to match your header images

Troubleshooting

Poor Hand Detection: Ensure adequate lighting and a clean background

Jerky Drawing: Adjust camera resolution or modify code for smoothing

Missing Header Images: Verify the correct path and presence of images

Performance Issues: Lower camera resolution or close resource-intensive applications

Last updated: March 17, 2025

