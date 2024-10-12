import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Définir les chemins vers les dossiers de données
train_dir = '/Users/amelievignes/Downloads/projet2/train'
validation_dir = '/Users/amelievignes/Downloads/projet2/validation'
test_dir = '/Users/amelievignes/Downloads/projet2/test'

# Préparation des données avec ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')  # Utiliser 'categorical' pour plusieurs classes

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

# Construire le modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Utiliser softmax pour plusieurs classes
])

# Compiler le modèle
model.compile(loss='binary_crossentropy',  # 'categorical_crossentropy' si plusieurs classes
              optimizer='adam',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Ajuster selon la taille du dataset
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

# Évaluation sur les données de test
test_loss, test_acc = model.evaluate(test_generator, steps=50)
print(f"Test accuracy: {test_acc}")

# Enregistrer le modèle au format .h5
model.save('/Users/amelievignes/Downloads/projet2/model.h5')