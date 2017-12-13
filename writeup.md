# Project: Vehicle Detection And Tracking

[//]: # (Image References)
[image1]: ./output_images/Car.png
[image2]: ./output_images/No Car.png
[image3]: ./output_images/Car Channel 0.png
[image4]: ./output_images/Car Channel 1.png
[image5]: ./output_images/Car Channel 2.png
[image6]: ./output_images/No Car Channel 0.png
[image7]: ./output_images/No Car Channel 1.png
[image8]: ./output_images/No Car Channel 2.png
[image9]: ./output_images/FirstDetection.png
[image10]: ./output_images/FinalDetection.png

### Pipeline

The resulting video can be found on YouTube by clicking on the image below:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/joKuHeSrCAo/0.jpg)](https://www.youtube.com/watch?v=joKuHeSrCAo)

The executable code can be found in: `main.py`

### Detection

#### 1. Feature extraction

The code for the feature extraction can be found within the file `feature.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Feature selection

I tried various combinations of parameters and...

#### 3. Classifier training

I trained a linear SVM using...

#### 4. Sliding window search

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 5. Detection resulits

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Tracking

#### 1. Data association for removing False Positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### 2. Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---


* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

