import cv2
import os
import numpy as np
from PIL import Image

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]     
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img, 'uint8')
        # This line is key: it balances the light so the computer isn't confused
        img_numpy = cv2.equalizeHist(img_numpy)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print("\n [INFO] Training... Please wait.")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
if not os.path.exists('trainer'): os.makedirs('trainer')
recognizer.write('trainer/trainer.yml') 
print(f"\n [SUCCESS] {len(np.unique(ids))} people trained. Now run detect.py")