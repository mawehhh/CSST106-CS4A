# Notebook Explanation: 4A_SANTOS_Exer4

This notebook covers various object detection techniques, showcasing both traditional and deep learning-based methods.

## 1. Download Libraries
The notebook begins by installing essential libraries for object detection:
- **TensorFlow** for deep learning models like SSD.
- **OpenCV** for image processing and traditional methods.
- **NumPy** for numerical operations.
- **Matplotlib** for visualizations.

```python
!pip install tensorflow
!pip install opencv-python
!pip install numpy
!pip install matplotlib
!pip install opencv-python-headless
```

## 2. Exercise 1: HOG (Histogram of Oriented Gradients) Object Detection
### Objective
Use HOG descriptors to detect objects based on gradients and edges, effective for detecting shapes like pedestrians.

### Explanation
1. Load and convert the image to grayscale.
2. Compute HOG descriptors to identify edges and visualize object outlines.

```python
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load an image and convert to grayscale
image = cv2.imread('pedestrian.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply HOG descriptor
features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)

# Display HOG image
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(hog_image, cmap='gray')
plt.show()
```

## 3. Exercise 2: YOLO (You Only Look Once) Object Detection
### Objective
Run real-time object detection using the YOLOv3 model.

### Explanation
1. Download YOLO weights, config, and COCO class labels.
2. Load the YOLO model, prepare the image, and detect objects.
3. Display bounding boxes and labels on the detected objects.

```python
# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Prepare and run detection
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Visualize detection
plt.imshow(yolo_result)
plt.axis('off')
plt.show()
```

## 4. Exercise 3: SSD (Single Shot MultiBox Detector) with TensorFlow
### Objective
Utilize SSD MobileNet V2 for efficient object detection.

### Explanation
1. Download and load SSD model and labels.
2. Run SSD model, processing detections.
3. Draw bounding boxes with labels and confidence scores.

```python
ssd_model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')

# Process SSD detections and display results
plt.imshow(ssd_result)
plt.show()
```

## 5. Exercise 4: Traditional vs. Deep Learning Object Detection Comparison
### Objective
Compare the HOG traditional approach with deep learning models (YOLO and SSD).

### Explanation
1. Display side-by-side images of HOG and deep learning detection results.
2. Highlight visual differences in detection accuracy.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
ax1.imshow(hog_image)
ax1.set_title('Traditional (HOG)')
ax1.axis('off')

ax2.imshow(deep_learning)
ax2.set_title('Deep Learning (YOLO + SSD)')
ax2.axis('off')
plt.tight_layout()
plt.show()
```
