import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import os
from dotenv import load_dotenv
load_dotenv()

IMG_HEIGHT = 30
IMG_WIDTH = 30
model_path = os.getenv("MODEL_PATH")
data_dir = os.getenv("DATASET_PATH")

def predict_traffic_sign(image_path, saved_model_path, data_directory):
    """
    Predicts the class of a single traffic sign image and displays
    the input image alongside the representative image for the predicted class.
    """
    # Load the class names mapping
    meta_df = pd.read_csv(os.path.join(data_directory, 'Meta.csv'))
    class_map = pd.Series(meta_df.SignId.values, index=meta_df.ClassId).to_dict()
    
    # Load the trained model
    loaded_model = tf.keras.models.load_model(saved_model_path)
    
    # Preprocess the input image
    img = PIL.Image.open(image_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_name = class_map[predicted_class]
    confidence = np.max(prediction)
    
    # Get the path to the representative class image
    meta_image_path = os.path.join(data_directory, 'Meta', f'{predicted_class}.png')
    
    # Display the input and predicted class images
    input_img = PIL.Image.open(image_path)
    meta_img = PIL.Image.open(meta_image_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Prediction Confidence: {confidence:.2%}", fontsize=16)
    
    ax1.imshow(input_img)
    ax1.set_title("Input Image")
    ax1.axis('off')

    ax2.imshow(meta_img)
    ax2.set_title(f"Predicted: '{predicted_name}'")
    ax2.axis('off')
    
    plt.show()
    
    return predicted_class, predicted_name, confidence

# Example usage:
sample_image_path = os.path.join(data_dir, "Test/00023.png")
predicted_class, name, conf = predict_traffic_sign(sample_image_path, model_path, data_dir)
print(f"Predicted Class ID: {predicted_class}, Name: {name}, Confidence: {conf:.4f}")