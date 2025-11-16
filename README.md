# 🎨 AI Air Canvas – Gesture Controlled Drawing App  
### Built with OpenCV + Mediapipe | By **Shivam Sahu**

AI Air Canvas is a gesture-controlled drawing application that lets you **draw in the air using your hands**.  
No touchscreen, no mouse — just your **index finger + webcam**.

The application detects your hand movements using **Mediapipe Hands**, processes gestures,  
and draws on a virtual canvas in real time.

---

## 🚀 Project Overview

This project uses **Computer Vision** to create a virtual drawing board controlled entirely through **hand gestures**:

- Draw using your **index finger**
- Erase using **index + middle fingers together**
- Change colors using a **thumb-up gesture**
- Pause drawing when making a **pinch gesture**
- Adjust brush size using **thumb-index distance**
- Clear the canvas with an **open-palm hold**
- Undo / Redo strokes
- Save artwork automatically with timestamps

This project showcases **gesture recognition**, **real-time tracking**,  
and fundamental **Human-Computer Interaction (HCI)** concepts.

---

## ✨ Features

### 👆 Drawing & Tracking
- Index finger detected as the brush
- Smooth drawing using point averaging
- Adjustable brush thickness

### ✋ Gesture Controls
| Gesture | Action |
|--------|--------|
| ☝ Index finger | Draw |
| ✌ Index + middle | Eraser |
| 🤏 Pinch (thumb + index) | Pause |
| 👍 Thumb up only | Change color |
| 🖐 Full open palm | Clear canvas |
| ✍ Move thumb closer/farther | Brush size |

### 🎨 Tools & UI
- 11-color palette  
- Undo / Redo  
- Save drawings with timestamps  
- Clean top control bar  
- Low-light enhancement using **CLAHE**  
- Works on any webcam

---

## 🖥 Demo

### 📸 **Screenshot**
(Add your screenshot here)
