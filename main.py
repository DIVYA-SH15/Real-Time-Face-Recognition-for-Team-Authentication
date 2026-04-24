import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os

# ----------------------------
# 1. LOAD MODEL & CLASSIFIER
# ----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Ensure the file exists before loading
if os.path.exists("trainer/trainer.yml"):
    recognizer.read("trainer/trainer.yml")
else:
    print("Error: 'trainer/trainer.yml' not found. Please ensure your training script has run successfully.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# CRITICAL: Ensure these IDs (0, 1) match exactly what was used in your training script
names = {0: "Divya", 1: "Punya"} 

# ----------------------------
# 2. CALIBRATED THRESHOLD
# ----------------------------
# LBPH Distance: 0 is perfect. 50-80 is typical for a good match. 
# Above 110 is usually a "Guess".
CONFIDENCE_THRESHOLD = 100 

# ----------------------------
# 3. GUI & CAMERA SETUP
# ----------------------------
cam = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Face Recognition Debugger")
root.geometry("800x600")

video_label = tk.Label(root)
video_label.pack(pady=10)

# Debug Info Panel
info_frame = tk.Frame(root, relief=tk.RIDGE, borderwidth=2)
info_frame.pack(fill=tk.X, padx=20, pady=10)

debug_text = tk.StringVar(value="Status: Waiting for face...")
status_label = tk.Label(info_frame, textvariable=debug_text, font=("Helvetica", 12, "bold"))
status_label.pack(pady=5)

dist_text = tk.StringVar(value="Raw Distance: N/A")
dist_label = tk.Label(info_frame, textvariable=dist_text, font=("Courier", 10))
dist_label.pack()

def update_frame():
    ret, frame = cam.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization improves recognition in varying light
    gray = cv2.equalizeHist(gray) 
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        debug_text.set("Status: Scanning... No face detected.")
        dist_text.set("Raw Distance: N/A")

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Predict returns [int Label, float Distance]
        label_id, distance = recognizer.predict(roi_gray)

        # Calculate a "Match Score" for display (0% to 100%)
        # Logic: if distance is 0, match is 100%. If distance is 100, match is 0%.
        match_perc = round(max(0, 100 - distance))
        
        dist_text.set(f"ID: {label_id} | Raw Distance: {round(distance, 2)} | Match: {match_perc}%")

        if distance < CONFIDENCE_THRESHOLD and label_id in names:
            name = names[label_id]
            color = (0, 255, 0) # Green
            debug_text.set(f"Status: MATCH FOUND - {name}")
        else:
            name = "Unknown"
            color = (0, 0, 255) # Red
            debug_text.set("Status: UNKNOWN PERSON")

        # Visual feedback on the video feed
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({match_perc}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert image for Tkinter
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(15, update_frame)

# Start logic
update_frame()
root.mainloop()

# Cleanup
cam.release()
cv2.destroyAllWindows()