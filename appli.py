import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('/Users/amelievignes/Downloads/projet2/model.h5')

# Fonction pour capturer une image depuis la caméra et prédire
def capture_and_predict():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="Erreur : Impossible d'ouvrir la caméra")
        return

    # Ajuster la luminosité et le contraste
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Ajuster selon tes préférences
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Ajuster selon tes préférences

    # Attendre 2-3 secondes pour que la caméra s'adapte à la lumière
    time.sleep(2)

    # Capturer une image
    ret, frame = cap.read()
    if not ret:
        result_label.config(text="Erreur : Impossible de capturer une image")
        cap.release()
        return

    # Convertir l'image de BGR à RGB (correction des couleurs pour Tkinter)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redimensionner l'image capturée à 128x128 pour la prédiction
    img_resized = cv2.resize(frame_rgb, (128, 128))

    # Convertir l'image en un tableau de numpy et normaliser
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Utiliser le modèle pour prédire
    prediction = model.predict(img_array)

    # Interpréter les résultats
    label = "Couteau" if prediction[0] > 0.5 else "Fourchette"

    # Afficher le résultat dans l'interface
    result_label.config(text=f'Prédiction : {label}')

    # Afficher l'image capturée dans la fenêtre Tkinter
    img = Image.fromarray(frame_rgb)  # Utilisation de l'image RGB pour Tkinter
    img = img.resize((300, 300))  # Redimensionner l'image pour l'affichage
    imgtk = ImageTk.PhotoImage(image=img)
    img_label.config(image=imgtk)
    img_label.image = imgtk  # Nécessaire pour garder une référence

    # Libérer la caméra
    cap.release()

# Interface graphique avec Tkinter
root = tk.Tk()
root.title("Reconnaissance Fourchette vs Couteau")

# Label pour afficher les résultats
result_label = Label(root, text="Appuyez sur le bouton pour prédire", font=("Helvetica", 16))
result_label.pack(pady=20)

# Bouton pour capturer une image et prédire
predict_button = tk.Button(root, text="Capturer et prédire", command=capture_and_predict, font=("Helvetica", 14))
predict_button.pack(pady=20)

# Label pour afficher l'image capturée
img_label = Label(root)
img_label.pack(pady=20)

# Lancer l'interface
root.mainloop()
