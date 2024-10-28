# Notebook Explanation: 4A_SANTOS_Exer3

This notebook demonstrates various computer vision exercises using OpenCV and related libraries. Each exercise involves a distinct technique to analyze or manipulate image data.

## 1. Downloading Libraries
The notebook begins by installing essential libraries:
- `opencv-python`: Primary library for image processing.
- `opencv-python-headless`: Supports OpenCV in environments without display capabilities.

```python
!pip install opencv-python opencv-python-headless
```

## 2. Exercise 1: Harris Corner Detection
### Objective
Detect and highlight corners within an image, which are areas of significant intensity change.

### Explanation
1. Load and convert the image to grayscale.
2. Apply Harris Corner Detection, which calculates intensity shifts to find potential corners.
3. Highlight the detected corners in red.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
image = cv2.imread('Pictureme.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
gray = np.float32(gray)
harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate and mark corners
harris_corners = cv2.dilate(harris_corners, None)
image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.show()
```

## 3. Exercise 2: HOG (Histogram of Oriented Gradients) Feature Extraction
### Objective
Extract gradient-based features commonly used in object detection.

### Explanation
1. Convert the image to grayscale.
2. Extract HOG features, which capture intensity gradients in cells.
3. Display the HOG image showing gradient orientation and magnitude.

```python
from skimage.feature import hog
from skimage import exposure

# Load and convert image
image = cv2.imread('Pictureme.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# HOG feature extraction
hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

# Rescale for visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('HOG Feature Extraction')
plt.show()
```

## 4. Exercise 3: FAST Keypoint Detection
### Objective
Detect keypoints using the FAST (Features from Accelerated Segment Test) algorithm, known for speed.

### Explanation
1. Convert the image to grayscale.
2. Detect keypoints with FAST.
3. Visualize the keypoints by marking them in blue.

```python
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))

# Display the result
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('FAST Keypoint Detection')
plt.show()
```

## 5. Exercise 4: Feature Matching using ORB and FLANN
### Objective
Match features between two images using ORB and FLANN, helpful in identifying correspondences.

### Explanation
1. Extract keypoints and descriptors for two images using ORB.
2. Use FLANN for efficient matching.
3. Filter good matches and display.

```python
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

plt.imshow(result)
plt.title('Feature Matching using ORB and FLANN')
plt.show()
```

## 6. Exercise 5: Image Segmentation using Watershed Algorithm
### Objective
Separate an image into distinct regions using the Watershed algorithm.

### Explanation
1. Convert to grayscale and apply binary thresholding.
2. Use morphological operations and distance transform to identify foreground.
3. Apply Watershed to mark boundaries, displayed in red.

```python
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

plt.imshow(boundary_img)
plt.title('Watershed Segmentation')
plt.show()
```
