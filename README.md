# Real-Time-Face-Recognition-for-Team-Authentication

A Python-based real-time face recognition system using OpenCV and LBPH (Local Binary Patterns Histogram) for identifying authorized team members.

## Features
- Detects and recognizes registered faces in real time
- Authorized members:
  - ID 0 → Divya
  - ID 1 → Punya
- Unknown persons are marked:
  - Red bounding box
  - Label: **Not in Team**
- Uses Haar Cascade for face detection
- Uses LBPH algorithm for face recognition

## Algorithm Used

### Face Detection
- Haar Cascade Classifier

### Face Recognition
- LBPH (Local Binary Pattern Histogram)

LBPH compares extracted facial features with trained samples and predicts identity with confidence.


