# CSST106-CS4A
**Exploring the Role of Computer Vision and Image Processing in AI**

https://github.com/user-attachments/assets/61bf6bbb-1acf-4a25-8f5d-8a5a9f1caafe

# Computer Vision Overview

Computer Vision is a subfield of Artificial Intelligence (AI) that enables machines to interpret and make decisions based on visual data, much like how humans perceive and understand images and videos. This involves the automatic extraction, analysis, and comprehension of information from visual inputs such as photographs, videos, and real-time camera feeds. The goal of computer vision is to simulate the human vision system, enabling machines to perform tasks like object detection, image classification, facial recognition, and scene reconstruction.

# Key Areas of Computer Vision

1. **Image Classification**: Assigning a label to an entire image based on its content. For example, identifying whether an image contains a cat or a dog.
2. **Object Detection**: Locating and identifying objects within an image. This includes drawing bounding boxes around detected objects.
3. **Image Segmentation**: Dividing an image into regions or segments that correspond to different objects or parts of objects. This can be used in applications like medical imaging where different tissues need to be distinguished.
4. **Facial Recognition**: Identifying or verifying a person from an image or video frame. This technology is widely used in security and authentication systems.
5. **Motion Analysis**: Tracking and analyzing the movement of objects in a sequence of images or video frames. This is crucial in video surveillance and sports analytics.
6. **3D Reconstruction**: Building a three-dimensional model of a scene or object from two-dimensional images. This is used in virtual reality, robotics, and autonomous vehicles.

# Role of Image Processing in AI Systems

Image processing is the foundational technology that powers computer vision. It involves the manipulation and analysis of images to improve their quality, extract valuable information, and prepare them for further processing by AI systems.

**The key roles of image processing in AI include:**

1. Preprocessing: Image processing techniques such as filtering, resizing, and normalization are used to enhance the quality of images and make them suitable for analysis. For instance, noise reduction and contrast enhancement help in improving the clarity of images. 
2. Feature Extraction: Image processing algorithms identify and extract important features from images, such as edges, textures, and shapes. These features are crucial for AI models to recognize patterns and make decisions.
3. Data Augmentation: In machine learning, data augmentation techniques like rotation, flipping, and scaling are applied to images to create variations and increase the size of the training dataset. This helps in building more robust AI models.
4. Transformation: Image processing is used to transform images into formats or domains that are more suitable for analysis. For example, converting an image from the spatial domain to the frequency domain using techniques like the Fourier transform.
5. Compression: Image processing techniques are used to compress images, reducing their size without significantly compromising quality. This is important for efficient storage and transmission of visual data in AI systems.
6. Object Detection and Recognition: Image processing plays a vital role in detecting and recognizing objects within images. Techniques like thresholding, edge detection, and contour detection help in isolating and identifying objects.

Computer Vision, powered by sophisticated image processing techniques, is at the heart of many AI systems, enabling machines to understand and interpret the visual world. From enhancing the quality of images to extracting meaningful features, image processing is critical in ensuring the accuracy and efficiency of AI models in various applications.

# Types of Image Processing Techniques

![image](https://github.com/user-attachments/assets/d30e336a-490d-4041-95e8-ee0c81b02d58)

# The three core techniques in image processing

## Image Enhancement
Image enhancement refers to the process of improving the visual appearance of an image or making it more suitable for specific applications. This technique involves modifying the intensity, contrast, sharpness, or color of an image to highlight important features and suppress irrelevant information.

**Example:**
Noise Reduction is a techniques like Gaussian blurring or median filtering reduce noise in images. For example, in low-light photography, noise reduction algorithms are used to produce clearer and more aesthetically pleasing images.

**Applications in AI**
A autonomous vehicles enhancing images by capturing using cameras helps in better detection of road signs, pedestrians, and other vehicles, improving the safety of self-driving cars.

## Image Segmentation
Image segmentation is the process of partitioning an image into distinct regions or objects that are meaningful and easier to analyze. It simplifies the representation of an image, making it easier to identify and analyze specific areas of interest.

**Example:**

Thresholding is a simple technique that converts a grayscale image into a binary image by setting all pixels above a certain intensity to white and below it to black. For example, in document scanning, thresholding is used to separate text from the background, making it easier for OCR (Optical Character Recognition) systems to read the text.

**Applications in AI**

Object Detection segmentation is used in AI models to identify and localize objects within an image. For instance, in autonomous vehicles, segmentation helps in distinguishing between the road, pedestrians, and other vehicles.

## Feature Extraction
Feature extraction involves identifying and isolating important attributes or characteristics from an image that are relevant for a specific task. These features could be edges, textures, shapes, or specific patterns that help in recognizing objects or scenes.

**Example:**

Edge detection techniques identify the boundaries of objects within an image. For example, in facial recognition systems, edge detection is used to identify the contours of facial features, such as eyes, nose, and mouth.

**Applications in AI**

Facial Recognition: Feature extraction is key to identifying unique facial features, enabling accurate recognition of individuals. This is widely used in security systems, smartphones, and social media platforms.

# Case Study Overview 

## Google Lens

* **Object Recognition and Detection:** It employs deep learning models like Convolutional Neural Networks (CNNs) to accurately identify and localize objects in images, making it effective for tasks like product identification and real-time object detection.
* **Text Recognition (OCR):** Google Lens uses Optical Character Recognition to extract text from images, even in challenging conditions, allowing users to copy, translate, or search for text quickly.
* **Image Enhancement:** Techniques like contrast enhancement and noise reduction improve image quality, enabling Google Lens to work effectively in poor lighting or with low-quality images.
* **Image Segmentation:** Segmentation techniques, such as semantic segmentation, help Google Lens isolate specific objects or regions in complex scenes, improving accuracy in identifying and analyzing multiple objects.
* **Feature Extraction:** Google Lens extracts and matches key features from images to recognize landmarks, products, and other specific items, providing accurate contextual information.
* **Real-Time Processing:** The app is optimized for real-time processing on mobile devices, allowing users to receive instant feedback as they interact with their environment.

## The Convolutional Neural Network (CNN) is the primary model used to address the simple problems related to computer vision in applications like Google Lens.

## Why CNN?

* Incorrect Object Recognition: Training on diverse datasets enhances CNNs' accuracy in challenging conditions.
* Low-Quality Images: CNNs can still extract important features from low-quality images, especially with techniques like super-resolution.
* Contextual Understanding: Advanced CNN architectures can interpret the broader context of images, improving overall understanding beyond individual objects.

# The Convolutional Neural Network (CNN) Illustration

![model drawio](https://github.com/user-attachments/assets/b82f5e57-a330-4d76-8d50-6b9831ecbf9c)

# Extension Activity

# Introduction to Novel Approaches in Image Processing: Neural Radiance Fields, NeRF Abstract
NeRF is a new kind of image synthesis or scene reconstruction that has been developed during the last years. It uses deep learning in order to create highly detailed 3D representations from 2D images, securing it as one of the most serious advances in image processing to date.

## Key Concepts
* **3D Scene Representation:** NeRF models the scene as a continuous volumetric field. Unlike other 3D representations that take mesh as input, NeRF represents both appearance and geometry of a scene with the help of a neural network.
* **Volumetric Rendering:** This technique synthesizes novel views of a scene by means of volume rendering-through the integration of colors and densities along the rays passing through a scene.
Training involves several 2D images from different viewpoints; the network learns to predict the color and density at any point in 3D space such that it can render photorealistic images from new viewpoints. * * * **Advantages High-quality 3D Reconstruction:** From a rather limited set of 2D images, NeRF may form highly detailed and photorealistic reconstructions.
It has the capability to capture intricate details in a scene, such as light effects or texture variations, beyond traditional 3D modeling techniques.
* **View synthesis:** It results in the creation of views of scenes that were not part of the original training data, enhancing applications related to virtual or augmented reality.
Potential Influence on Future AI Systems
* **Improved Virtual and Augmented Reality:** NeRF can transform the experience of VR and AR into immersive, real environments by allowing the construction of highly detailed 3D models from a limited set of inputs.
* **Advanced Robotics and Navigation:** Robots will gain a much better level of understanding of complex environments by building and interpreting high-fidelity 3D maps.
* **Improved Image and Video Editing:** NeRF could allow for advanced editing and special effects in images and videos, opening more creativity and visual possibility in media production.
* **Medical Imaging:** NeRF can be used in health to derive detailed 3D models from 2D medical scans, thus enhancing diagnosis and surgical planning activities.
Challenges and Future Directions
* **Computational Cost:** The biggest limitation in most current NeRF works involves high computational resources and time for training, which may narrow its applications in real-world practice.
* **Limitations:** High-quality results depend on a number of large image datasets, not always available; Real-time Processing: NeRF is somewhat latent; hence, adapting NeRF to real-time processing is considered one of the open lines of research. Conclusion
Neural Radiance Fields represent a huge leap forward in the handling and reconstruction of images and 3D scenes. Their applications may span from education to entertainment and promise to make technology both more practical and creative. For the technique to fully develop, surmounting its limitations will be key to wide acceptance and, therefore, impact.

# Implementation

## Input Image

This is the initial data fed into the model. For image classification, this would be a 2D image with specific dimensions (e.g., 150x150 pixels) and color channels (e.g., RGB with 3 channels).

## Convolutional Layer

This layer applies a set of filters to the input image to create feature maps. It detects patterns such as edges, textures, etc.

`Conv2D(32, (3, 3), activation='relu')`

## Activation Layer (ReLU)

This introduces non-linearity into the model which helps it learn complex patterns. ReLU is a popular activation function that outputs the input directly if it is positive; otherwise, it outputs zero.

The ReLU activation function is applied directly within the `Conv2D` layer by specifying `activation='relu'`.

## Pooling Layer

This layer reduces the spatial dimensions of the feature maps to decrease the number of parameters and computation. It helps in making the model invariant to small translations in the image.

`MaxPooling2D(pool_size=(2, 2))`
`pool_size=(2, 2)` specifies the size of the pooling window (2x2 in this case).

## Flattening Layer

This layer flattens the 2D feature maps into a 1D vector. This is necessary before passing the data to fully connected (dense) layers.

`Flatten()`

## Fully Connected Layer

These layers are dense layers that connect every neuron from the previous layer to every neuron in the current layer. They are used to learn complex representations and patterns.

`Dense(512, activation='relu')`

`512` specifies the number of neurons in this layer.

`activation='relu'` applies the ReLU activation function.

## Output Layer (Softmax)

This layer outputs the probability distribution over classes. Softmax activation converts the raw scores into probabilities.

`Dense(len(train_generator.class_indices), activation='softmax')`

`len(train_generator.class_indices)` determines the number of output classes

`activation='softmax'` converts logits into probabilities.

## Predicted Class

This is the final class prediction made by the model after training.

#Full Code

import tensorflow as tf
`from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import os`

# Set dataset paths (adjust these paths to your local setup)
`train_dir = r'C:\Users\asus\CNN\MY_data\train'
test_dir = r'C:\Users\asus\CNN\MY_data\test'`

# Verify number of images
`def count_images_in_directory(directory):
    class_counts = {}
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            class_counts[subdir] = len(os.listdir(subdir_path))
    return class_counts`

`train_counts = count_images_in_directory(train_dir)
test_counts = count_images_in_directory(test_dir)`

`print(f'Train directory counts: {train_counts}')
print(f'Test directory counts: {test_counts}')`

# Set up data augmentation and normalization
`train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)`

# Create data generators
`train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True  # Shuffle to ensure diversity in batches
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # No shuffle for test data
)`

# Build the CNN model
`model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),`

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Number of classes dynamically
`])`

# Compile the model
`model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)`

# Train the model
`history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)`

# Evaluate the model
`test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')`

# Save the model
`model.save(r'C:\Users\asus\CNN\Classifier.h5')`

# Predicting

`import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model`

# Load the trained model
`model = load_model(r'C:\Users\asus\CNN\Classifier.h5')`

# Function to prepare an image for prediction
`def prepare_image(image_path, target_size=(150, 150)):
    # Load the image
    img = load_img(image_path, target_size=target_size)`
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes for cleaner display
    plt.show()

    # Convert the image to array
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Prepare an image for prediction and display it
`img_array = prepare_image(r'C:\Users\asus\CNN\prots.jpg')`

# Predict the class
`predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)`

# Assuming you have access to `train_generator` for class names
`class_names = list(train_generator.class_indices.keys())  # Ensure train_generator is defined earlier
print(f'Predicted class: {class_names[predicted_class[0]]}')`

#Example Output

![image](https://github.com/user-attachments/assets/9f98361f-1590-47b8-9b5e-d6fbab28f8f1)


































































