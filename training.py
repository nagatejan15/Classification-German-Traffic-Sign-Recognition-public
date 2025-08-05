import opendatasets as od
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import PIL
import kagglehub
import shutil
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

data_dir = os.getenv("DATASET_PATH")
train_path = os.path.join(data_dir, 'Train')
test_csv_path = os.path.join(data_dir, 'Test.csv')
meta_csv_path = os.path.join(data_dir, 'Meta.csv')
model_path = os.getenv("MODEL_PATH")

IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CATEGORIES = 43

def train_and_save_classification():
    images = []
    labels = []
    for i in range(NUM_CATEGORIES):
        path = os.path.join(train_path, str(i))
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            image = PIL.Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
            images.append(np.array(image))
            labels.append(i)

    images = np.array(images)
    labels = np.array(labels)

    images = images / 255.0
    labels = to_categorical(labels, NUM_CATEGORIES)

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Model Architecture 
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile and Train Model 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(" Starting Model Training ")
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
    print(" Model Training Finished ")

    # Evaluation 
    test_data = pd.read_csv(test_csv_path)
    y_test = test_data['ClassId'].values
    test_img_paths = [os.path.join(data_dir, p) for p in test_data['Path'].values]

    test_images = []
    for path in test_img_paths:
        image = PIL.Image.open(path).resize((IMG_WIDTH, IMG_HEIGHT))
        test_images.append(np.array(image))

    X_test = np.array(test_images) / 255.0
    y_test_cat = to_categorical(y_test, NUM_CATEGORIES)

    loss, accuracy = model.evaluate(X_test, y_test_cat)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save Model 
    model.save(model_path)
    print(f"Model saved to {model_path}")
