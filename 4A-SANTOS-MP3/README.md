
# Python Image Processing Script

## Overview
This script utilizes OpenCV for various image processing tasks such as keypoint detection, feature extraction, and image alignment.

## Prerequisites
- Python 3.x
- OpenCV
- Matplotlib

## Installation
Update and install the necessary packages:

```bash
apt-get update
apt-get install -y cmake build-essential pkg-config
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
mkdir -p opencv/build
cd opencv/build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF ..
make -j8
make install
```

## Usage
Replace the paths `/content/rose.jpg` and `/content/daisy.jpg` with your actual image paths before running the script.

## Steps
1. **Load Images:** Loads two images for processing.
2. **Feature Detection and Description:** Uses SIFT, SURF, and ORB algorithms.
3. **Feature Matching:** Employs Brute-Force and FLANN-based methods for matching.
4. **Image Alignment:** Utilizes Homography to align two images.

## Output
Generates images with keypoints, feature matches, and the aligned image. These outputs are saved locally.

## Visualization
View keypoints and matches via Matplotlib plots directly in the script execution.

