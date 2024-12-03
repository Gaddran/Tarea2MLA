#!/usr/bin/env python
# coding: utf-8

import math
import pandas as pd
import numpy as np
import random
import sklearn.neural_network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers, regularizers
from keras.utils import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


# Paths to dataset (update with your actual paths)
train_dir = r"/mnt/c/Users/Giuseppe/My documents/github/tarea2mla/data/train"
test_dir = r"/mnt/c/Users/Giuseppe/My documents/github/tarea2mla/data/test"


get_ipython().run_line_magic('pwd', '')


#parametros
activation = 'relu'  # Activación para las capas ocultas
learning_rate = 0.01  # Tasa de aprendizaje
batch_size = 16  # Tamaño de lote
epochs = 10  # Número de épocas
img_size = (150, 150)


train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=img_size,
    batch_size=batch_size
)


class_names = dataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")


normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))


# Define the model
model = models.Sequential([
    # Explicit Input layer
    layers.Input(shape=(img_size[0], img_size[1], 3)),
    
    # Rescaling layer
    layers.Rescaling(1./255),
    
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display summary
model.summary()


# Get the total number of batches in the dataset
total_batches = tf.data.experimental.cardinality(dataset).numpy()

# Calculate the number of training and validation batches
train_batches = int(total_batches * 0.8)  # Convert to integer
val_batches = total_batches - train_batches  # Remaining for validation

# Split the dataset
train_dataset = dataset.take(train_batches)
val_dataset = dataset.skip(train_batches)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)


history = model.fit(
    train_dataset,  # Training data
    validation_data=val_dataset,  # Validation data
    epochs=10,  # Adjust number of epochs
    verbose=1  # Shows training progress
)


#Crear el modelo
model = models.Sequential()

# Primera capa: Aplanar las imágenes
model.add(layers.Flatten(input_shape=(img_size[0], img_size[1], 3)))  # Aplanar las imágenes

# Segunda capa oculta: 5 nodos
model.add(layers.Dense(units=5, activation=activation))

# Tercera capa oculta: 2 nodos
model.add(layers.Dense(units=2, activation=activation))

# Cuarta capa oculta adicional: 32 nodos
model.add(layers.Dense(units=32, activation=activation))

# Quinta capa oculta adicional: 16 nodos
model.add(layers.Dense(units=16, activation=activation))

# Sexta capa oculta adicional: 8 nodos
model.add(layers.Dense(units=8, activation=activation))

# Capa de salida: 1 nodo con activación 'sigmoid' (clasificación binaria)
model.add(layers.Dense(units=1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.SGD(learning_rate=learning_rate),
             metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_data, epochs=epochs, batch_size=batch_size)


val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")


import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# Evaluar el modelo
train_acc = model.evaluate(train_data, batch_size=batch_size)[1]
test_acc = model.evaluate(test_data, batch_size=batch_size)[1]
print('Training accuracy: %s' % train_acc)
print('Testing accuracy: %s' % test_acc)

# Graficar las pérdidas
losses = history.history['loss']
plt.plot(range(len(losses)), losses, 'r')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el entrenamiento')
plt.show()


image_path1 = "/mnt/c/Users/Giuseppe/My documents/github/tarea2mla/data/test/muffin/img_2_8.jpg"
image_path2 = "/mnt/c/Users/Giuseppe/My documents/github/tarea2mla/data/test/muffin/img_3_373.jpg"

image_path3 = "/mnt/c/Users/Giuseppe/My documents/github/tarea2mla/data/test/chihuahua/img_1_350.jpg"
image_path4 = "/mnt/c/Users/Giuseppe/My documents/github/tarea2mla/data/test/chihuahua/img_1_918.jpg"


def preprocess_image(image_path, img_size=img_size):
    #Cargar y procesar la imagen
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalizar
    return img_array[np.newaxis, ...]  # Expandir dimensiones

def classify_image(model, image_array):
    #predicción
    prediction = model.predict(image_array)[0][0]
    result = "Chihuahua" if prediction > 0.5 else "Muffin"
    confidence = max(prediction, 1 - prediction)
    return result, confidence


img_array1 = preprocess_image(image_path1)
img_array2 = preprocess_image(image_path2)
img_array3 = preprocess_image(image_path3)
img_array4 = preprocess_image(image_path4)

# Clasificar las imágenes
result1, confidence1 = classify_image(model, img_array1)
result2, confidence2 = classify_image(model, img_array2)
result3, confidence3 = classify_image(model, img_array3)
result4, confidence4 = classify_image(model, img_array4)

# Imprimir los resultados
print(f"La imagen 1 fue clasificada como: {result1} (confianza: {confidence1:.2f})")
print(f"La imagen 2 fue clasificada como: {result2} (confianza: {confidence2:.2f})")
print(f"La imagen 3(chihuaha) fue clasificada como: {result3} (confianza: {confidence3:.2f})")
print(f"La imagen 4(chihuaha) fue clasificada como: {result4} (confianza: {confidence4:.2f})")


# Crear el modelo convolucional
model_cnn = models.Sequential()

# Primera capa convolucional
model_cnn.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation=activation, input_shape=(500, 500, 3)))
model_cnn.add(layers.MaxPooling2D(pool_size=(2, 2))) 

# Segunda capa convolucional
model_cnn.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation))
model_cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Aplanar para pasar a la parte densa
model_cnn.add(layers.Flatten())

# Una sola capa densa
model_cnn.add(layers.Dense(units=64, activation=activation))

# Capa de salida
model_cnn.add(layers.Dense(units=1, activation='sigmoid'))

# Compilar el modelo
model_cnn.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),  # Tasa de aprendizaje predeterminada
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


history_cnn = model_cnn.fit(train_data, epochs=epochs, batch_size=batch_size)

