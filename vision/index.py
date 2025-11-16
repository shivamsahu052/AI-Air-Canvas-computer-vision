"""
Air Canvas - Gesture Controlled Drawing
Author: Shivam Sahu

A gesture‑controlled drawing application using Python, OpenCV, and Mediapipe.
Draw, erase, change colors, adjust brush size, and clear the canvas using only
hand gestures. The webcam tracks the index finger for drawing.

Features:
- Index‑finger drawing
- Eraser (index + middle fingers together)
- Pause/Resume using pinch gesture
- Color switching using thumb‑up gesture
- Brush thickness by thumb‑index distance
- Undo / Redo
- Save drawing with timestamp
- Low‑light improvement using CLAHE
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
from datetime import datetime
import math

# ---------------- Configuration ----------------
CAM_INDEX = 0  # Change to 1 or 2 if webcam doesn't open
FRAME_W, FRAME_H = 960, 720
BG_COLOR = (255, 255, 255)
SMOOTHING_WINDOW = 4
ERASER_RADIUS = 40
UNDO_LIMIT = 20

PALETTE = [
    (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (128, 0, 128), (0, 128, 255), (50, 50, 50), (200, 200, 200)
]
color_names = ['Black','Blue','Green','Red','Yellow','Magenta','Cyan','Purple','Orange','DarkGray','LightGray']
colorIndex = 1

# ---------------- Utilities ----------------
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ---------------- Mediapipe ----------------
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ---------------- State ----------------
paintWindow = np.ones((FRAME_H, FRAME_W, 3), dtype=np.uint8) * 255
undo_stack = []
redo_stack = []
strokes = [[] for _ in PALETTE]
current_stroke = None
raw_points = deque(maxlen=SMOOTHING_WINDOW)
mode = 'PAUSE'
last_clear_time = 0
clear_hold_sec = 1.2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ---------------- Helper Functions ----------------
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def smooth_point(points):
    if not points:
        return None
    return (
        int(sum(p[0] for p in points) / len(points)),
        int(sum(p[1] for p in points) / len(points))
    )

def push_undo():
    if len(undo_stack) >= UNDO_LIMIT:
        undo_stack.pop(0)
    undo_stack.append(paintWindow.copy())
    redo_stack.clear()

def do_undo():
    global paintWindow
    if undo_stack:
        redo_stack.append(paintWindow.copy())
        paintWindow = undo_stack.pop()

def do_redo():
    global paintWindow
    if redo_stack:
        undo_stack.append(paintWindow.copy())
        paintWindow = redo_stack.pop()

def save_drawing(img):
    ensure_folder('saved_drawings')
    filename = datetime.now().strftime('saved_drawings/drawing_%Y%m%d_%H%M%S.png')
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")

# Finger detection

def fingers_up(lm):
    tips = [4, 8, 12, 16, 20]
    pip = [2, 6, 10, 14, 18]
    out = []

    # Thumb
    out.append(lm[tips[0]][0] < lm[pip[0]][0])

    # Other fingers
    for i in range(1, 5):
        out.append(lm[tips[i]][1] < lm[pip[i]][1])
    return out

# UI top bar

def draw_top_bar(img):
    cv2.rectangle(img, (0,0), (FRAME_W,70), (230,230,230), -1)

    buttons = ['CLEAR (hold)', 'UNDO (z)', 'REDO (y)', 'SAVE (s)', 'COLOR (c)']
    for i, b in enumerate(buttons):
        x = 10 + i*190
        cv2.rectangle(img, (x,10), (x+180,60), (200,200,200), -1)
        cv2.putText(img, b, (x+10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40,40,40),2)

    cv2.putText(img, f"Mode: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50),2)
    cv2.putText(img, f"Color: {color_names[colorIndex]}", (260, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PALETTE[colorIndex],2)

# ---------------- Main Loop ----------------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("Running Air Canvas... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected. Try changing CAM_INDEX.")
        break

    frame = cv2.flip(frame, 1)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = clahe.apply(L)
    frame_proc = cv2.cvtColor(cv2.merge((L,A,B)), cv2.COLOR_LAB2BGR)

    results = hands.process(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB))

    draw_top_bar(frame)
    index_tip = None

    if results.multi_hand_landmarks:
        for handslms in results.multi_hand_landmarks:
            lm = [(int(l.x * FRAME_W), int(l.y * FRAME_H)) for l in handslms.landmark]
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fingers = fingers_up(lm)
            index_tip = lm[8]
            thumb = lm[4]
            middle = lm[12]

            d_thumb_idx = dist(thumb, index_tip)
            d_idx_mid = dist(index_tip, middle)

            erasing = d_idx_mid < 40

            if d_thumb_idx > 45:
                mode = 'ERASE' if erasing else 'DRAW'
            else:
                mode = 'PAUSE'

            if fingers[0] and not any(fingers[1:]):
                colorIndex = (colorIndex + 1) % len(PALETTE)
                time.sleep(0.3)

            if all(fingers):
                if last_clear_time == 0:
                    last_clear_time = time.time()
                elif time.time() - last_clear_time > clear_hold_sec:
                    push_undo()
                    paintWindow[:] = BG_COLOR
                    last_clear_time = 0
            else:
                last_clear_time = 0

            brush_size = int(np.interp(d_thumb_idx, [20,200], [2,60]))

            if mode == 'DRAW' and not erasing:
                raw_points.append(index_tip)
                p = smooth_point(list(raw_points))
                if p:
                    if current_stroke is None:
                        push_undo()
                        current_stroke = {'color': colorIndex, 'pts': [], 'size': brush_size}
                    current_stroke['pts'].append((p, brush_size))

            elif mode == 'ERASE':
                push_undo()
                cv2.circle(paintWindow, index_tip, ERASER_RADIUS, BG_COLOR, -1)

            else:
                if current_stroke is not None:
                    strokes[current_stroke['color']].append(current_stroke)
                    pts = current_stroke['pts']
                    for i in range(1, len(pts)):
                        cv2.line(paintWindow, pts[i-1][0], pts[i][0], PALETTE[current_stroke['color']], pts[i][1])
                    current_stroke = None
                    raw_points.clear()

    else:
        # If no hand is detected, finalize stroke if it exists
        if current_stroke is not None:
            strokes[current_stroke['color']].append(current_stroke)
            pts = current_stroke['pts']
            for i in range(1, len(pts)):
                cv2.line(paintWindow, pts[i-1][0], pts[i][0],
                         PALETTE[current_stroke['color']], pts[i][1])
            current_stroke = None
            raw_points.clear()

    # ---------------- Display ----------------
    canvas_display = paintWindow.copy()
    if current_stroke is not None:
        pts = current_stroke['pts']
        for i in range(1, len(pts)):
            cv2.line(canvas_display, pts[i-1][0], pts[i][0],
                     PALETTE[current_stroke['color']], pts[i][1])

    cv2.imshow("Canvas", canvas_display)
    cv2.imshow("Air Canvas - Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        do_undo()
    elif key == ord('y'):
        do_redo()
    elif key == ord('c'):
        colorIndex = (colorIndex + 1) % len(PALETTE)
    elif key == ord('s'):
        save_drawing(paintWindow)

cap.release()
cv2.destroyAllWindows()
print("Exited.")

