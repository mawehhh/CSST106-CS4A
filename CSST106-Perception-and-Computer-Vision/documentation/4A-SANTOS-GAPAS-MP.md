
# YOLOv8 Implementation Documentation

This documentation provides a detailed guide on using YOLOv8 for object detection, specifically for detecting and recognizing the six sides of a die. The steps include installation, data preparation, model setup, training, and evaluation.

---

## 1. Installation and Setup

**Installing YOLOv8 (Ultralytics Package)**:
To implement YOLOv8, first, install the `ultralytics` package, which provides access to the latest YOLOv8 model and functionalities.

```python
!pip install ultralytics
```

**Importing Required Libraries**:
After installation, import the necessary libraries for handling images, visualizing results, and using YOLOv8. Ensure to import YOLO from the `ultralytics` package to initialize and configure the YOLOv8 model.

```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# Check system and version compatibility for YOLOv8
ultralytics.checks()
```

---

## 2. Dataset Preparation

To use YOLOv8 effectively, the dataset needs to be in a format compatible with object detection, including labeled images for training, validation, and testing.

**Extracting Data**:
If the dataset is in a ZIP file, extract it to a specified path. This example shows how to extract the dataset for easier access and organization.

```python
from zipfile import ZipFile

# Define paths to your ZIP file and extraction directory
zip_path = '/content/drive/MyDrive/Midterm/dice6_side_yolov8.zip'
extract_path = '/content/drive/MyDrive/Midterm'

# Extracting the dataset
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extraction completed!")
```

**Defining Dataset Paths**:
Specify paths for training, validation, and testing images and labels. This organization helps YOLOv8 locate and differentiate between different datasets during training and evaluation.

```python
# Paths for training, validation, and testing datasets
train_images_path = '/content/drive/MyDrive/Midterm/train/images'
train_labels_path = '/content/drive/MyDrive/Midterm/train/labels'

valid_images_path = '/content/drive/MyDrive/Midterm/valid/images'
valid_labels_path = '/content/drive/MyDrive/Midterm/valid/labels'

test_images_path = '/content/drive/MyDrive/Midterm/test/images'
test_labels_path = '/content/drive/MyDrive/Midterm/test/labels'
```

---

## 3. Loading Images and Labels

For YOLOv8, each image requires a corresponding label file containing annotations in YOLO format. The function below helps load an image and its associated labels.

```python
def load_image_and_label(image_folder, label_folder, filename):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    label_filename = filename.replace('.jpg', '.txt')
    label_path = os.path.join(label_folder, label_filename)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    return image, labels

# Example of loading and displaying an image with labels
image, labels = load_image_and_label(train_images_path, train_labels_path, 'example.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()
print("Labels:", labels)
```

This function facilitates verification of images and labels before training, helping to ensure data integrity.

---

## 4. Image Preprocessing

Preprocessing steps standardize image size and pixel values to match the YOLOv8 model requirements. This step includes resizing images and normalizing pixel values, which helps improve model training.

```python
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found at {image_path}")
        return None
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    return image
```

This function helps maintain consistent input dimensions, enhancing model performance by standardizing the inputs.

---

## 5. YOLOv8 Model Setup and Training

**Initializing the YOLOv8 Model**:
Initialize YOLOv8 using a configuration file specifying the model size and settings (e.g., `yolov8n.yaml`). YOLOv8 offers a range of model sizes (nano, small, medium, etc.) suited to different performance requirements.

```python
# Initialize YOLO model for training on custom classes
model = YOLO('yolov8n.yaml')  # Choose model size as per requirements (e.g., yolov8s for small)
```

**Training the Model**:
YOLOv8 can be trained using custom settings such as dataset path, number of epochs, and batch size. In this example, the model is trained on 100 epochs with a batch size of 16.

```python
model.train(data='/content/drive/MyDrive/Midterm/dice6_side_yolov8.yaml', epochs=100, batch=16)
```

The `train()` function handles dataset loading, augmentation, and training, streamlining the model training process.

---

## 6. Evaluation and Testing

Once trained, the model can be evaluated on the test dataset. YOLOv8’s `val()` function calculates metrics like precision, recall, and mean Average Precision (mAP) to assess model performance.

```python
# Evaluate the model on test data
results = model.val()
print("Evaluation results:", results)
```

These metrics help determine the effectiveness of the model and indicate areas where further fine-tuning may be required.

---

## Summary

This documentation provides a structured approach to implementing YOLOv8 for object detection. Each step covers essential processes, from installation and dataset preparation to model training and evaluation, ensuring a complete workflow for effective object detection using YOLOv8.

