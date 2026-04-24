import cv2
import os

# --- SETTINGS ---
face_id = 1  # Change this to 1 for Punya, 2 for someone else, etc.
count = 0
max_images = 100
dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"\n [INFO] Initializing face capture. Look at the camera and wait...")

while True:
    ret, img = cam.read()
    if not ret: break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        # Crop the face and resize it to a standard 200x200
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        
        # Save the image in the format: User.[ID].[SampleNumber].jpg
        file_name = f"{dataset_path}/User.{face_id}.{count}.jpg"
        cv2.imwrite(file_name, face_img)

        # Draw a rectangle so you know it's working
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"Captured: {count}/{max_images}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Capturing Face Data', img)

    # Stop if ESC is pressed or we reach the limit
    if cv2.waitKey(100) & 0xFF == 27 or count >= max_images:
        break

print(f"\n [SUCCESS] {count} images saved for ID {face_id} in /{dataset_path}")
cam.release()
cv2.destroyAllWindows()