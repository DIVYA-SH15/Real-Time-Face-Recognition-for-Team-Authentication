import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from collections import deque

# 1. SETUP
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# CONFIGURATION
names = {1: "Punya", 0: "Divya"}
CONFIDENCE_THRESHOLD = 75  # Lowered to 75 for extreme strictness
STABILITY_BUFFER = deque(maxlen=20) # Must be consistent for 20 frames

window = tk.Tk()
window.title("Team Access System Pro")
video_label = tk.Label(window)
video_label.pack()

caption_label = tk.Label(window, text="Scanning...", font=("Arial", 22, "bold"))
caption_label.pack(pady=20)

cam = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cam.read()
    if not ret: return

    # --- ADVANCED PRE-PROCESSING ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE improves local contrast (makes eyes/nose/mouth details pop)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Increasing minNeighbors to 10 ensures we only detect REAL faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(120, 120))

    current_name = "Unknown"
    is_team = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_detected, distance = recognizer.predict(roi_gray)

        # Accuracy Check
        if distance < CONFIDENCE_THRESHOLD:
            STABILITY_BUFFER.append(id_detected)
        else:
            STABILITY_BUFFER.append(-1)

        # Voting Logic
        stable_id = max(set(STABILITY_BUFFER), key=list(STABILITY_BUFFER).count)

        if stable_id in names:
            current_name = names[stable_id]
            is_team = True
            color = (0, 255, 0)
        else:
            current_name = "Unknown"
            is_team = False
            color = (0, 0, 255)

        # UI Overlay
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{current_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- FINAL CAPTION LOGIC ---
    if is_team:
        caption_label.config(text=f"YOU ARE IN TEAM: {current_name}", fg="green")
    elif len(faces) > 0:
        caption_label.config(text="YOU ARE NOT IN TEAM: Unknown", fg="red")
    else:
        caption_label.config(text="Scanning...", fg="blue")

    # Render
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img))
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)
    window.after(10, update_frame)

update_frame()
window.mainloop()
cam.release()
cv2.destroyAllWindows()