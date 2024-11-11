import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
import streamlit as st
import tempfile

# Initialiser le modèle et les détecteurs
detector = HandDetector(maxHands=1)

# Charger le modèle et compiler
model = tf.keras.models.load_model("Model3/hand_sign_detection_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

classifier = Classifier("Model3/hand_sign_detection_model.h5", "Model3/labels.txt")
offset = 20
imgSize = 300
labels = ["Mal de tete", "Au Secours", "Vertige", "Vomissements"]

# Fonction de redimensionnement de l'image
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def main():
    st.title('Détection de Langage des Signes')

    st.sidebar.title('Paramètres de Détection')
    use_webcam = st.sidebar.button('Utiliser la Webcam')
    video_file_buffer = st.sidebar.file_uploader("Téléchargez une vidéo", type=["mp4", "mov", "avi", "asf", "m4v"])
    confidence_threshold = st.sidebar.slider('Seuil de Confiance', min_value=0.0, max_value=1.0, value=0.5)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            st.write("Veuillez utiliser la webcam ou télécharger une vidéo.")
            return
    else:
        tfflie = tempfile.NamedTemporaryFile(delete=False)
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    stframe = st.empty()
    stop_button = st.sidebar.button('Arrêter la Détection')

    if stop_button:
        st.stop()

    while True:
        success, img = vid.read()
        if not success:
            st.write("Erreur: Impossible de lire la vidéo.")
            break
        
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                
                if 0 <= index < len(labels):
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                  (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (255, 0, 255), 4)

                    stframe.image(imgOutput, channels="BGR", use_column_width=True)
                else:
                    st.write(f"Erreur: index de prédiction {index} hors des limites pour les labels.")
        
    vid.release()

if __name__ == '__main__':
    main()
