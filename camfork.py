import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np
import time

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('/Users/amelievignes/Downloads/projet2/model.h5')

# Fonction pour capturer une image depuis la caméra
def capture_and_predict(model):
    # Ouvrir la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra")
        return

    # Ajuster la luminosité et le contraste
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.5)

    # Boucle pour capturer des images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de capturer une image")
            break

        # Redimensionner l'image capturée à 128x128
        img_resized = cv2.resize(frame, (128, 128))

        # Convertir l'image en un tableau de numpy et normaliser
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Utiliser le modèle pour prédire
        prediction = model.predict(img_array)

        # Interpréter les résultats
        if prediction[0] > 0.5:
            label = "Couteau"
        else:
            label = "Fourchette"

        # Afficher le label sur l'image
        cv2.putText(frame, f'Prédiction: {label}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Afficher l'image capturée avec la prédiction
        cv2.imshow('Caméra', frame)

        # Attendre 1 seconde avant de capturer la prochaine image
        time.sleep(1)

        # Quitter si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la caméra et fermer toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

# Appeler la fonction
capture_and_predict(model)
