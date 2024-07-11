import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

def train_and_save_model():
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images / 255.0, testing_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    model.save('image_classifier.model')

if __name__ == "__main__":
    train_and_save_model()
