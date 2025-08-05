# **German Traffic Sign Recognition**

This project is a deep learning model for classifying German traffic signs. It uses a Convolutional Neural Network (CNN) to identify the traffic signs from images. The model is built with TensorFlow and Keras.

## **Project Overview**

This repository contains the following files:

* training.py: A script to train the traffic sign classification model.  
* classification.py: A script to predict the class of a single traffic sign image using the trained model.  
* requirements.txt: A file listing the project's dependencies.  
* gtsrb_cnn_model.keras: The saved, pre-trained model file.

## **Features**

* Builds and trains a CNN model for traffic sign classification.  
* The model architecture consists of two convolutional layers, two max-pooling layers, and two dense layers.  
* Uses dropout for regularization.  
* Evaluates the model's accuracy on a test set.  
* Saves the trained model to a file.  
* Provides a script to load the saved model and make predictions on new images.  
* Displays the input image and a representative image of the predicted class with the confidence score.

## **Dataset**

The model is trained on the [German Traffic Sign Recognition Benchmark (GTSRB) dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). You will need to download the dataset from this link and place it in the appropriate directory.

## **Getting Started**

### **Prerequisites**

You need to have Python and the following libraries installed. You can install them using pip:
```
pip install -r requirements.txt
```

### **Setup**

1. **Clone the repository:**  
   ```
   git clone <repository-url>  
   cd <repository-directory>
   ```

2. Download the dataset:  
   Download the dataset from the link provided above and extract it.  
3. Environment Variables:  
   Create a .env file in the root directory and add the following paths:
   ```  
   DATASET_PATH="path/to/your/dataset"  
   MODEL_PATH="path/to/save/your/model/gtsrb_cnn_model.keras"
   ```

### **Training the Model**

To train the model, run the training.py script:

```
python training.py
```

This will train the model on the GTSRB dataset and save the trained model as gtsrb\_cnn\_model.keras.

### **Making Predictions**

To make a prediction on a new traffic sign image, use the classification.py script. Make sure to update the sample\_image\_path in the script to the path of your image.

```
python classification.py
```

This will load the pre-trained model, predict the traffic sign, and display the input image along with the predicted traffic sign and the confidence of the prediction.

## **Model Architecture**

The CNN model architecture is defined in training.py and is as follows:

* **Conv2D Layer:** 32 filters, (5, 5\) kernel, 'relu' activation  
* **MaxPool2D Layer:** (2, 2\) pool size  
* **Dropout Layer:** 0.25 dropout rate  
* **Conv2D Layer:** 64 filters, (3, 3\) kernel, 'relu' activation  
* **MaxPool2D Layer:** (2, 2\) pool size  
* **Dropout Layer:** 0.25 dropout rate  
* **Flatten Layer**  
* **Dense Layer:** 256 units, 'relu' activation  
* **Dropout Layer:** 0.5 dropout rate  
* **Dense Layer (Output):** 43 units (for 43 classes), 'softmax' activation

The model is compiled using the 'adam' optimizer and 'categorical\_crossentropy' as the loss function.