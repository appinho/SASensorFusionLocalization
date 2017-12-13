# Project: Vehicle Detection And Tracking

[//]: # (Image References)

[image1]: ./output_images/Car.png "Car"
[image2]: ./output_images/No_Car.png "No Car"
[image3]: ./output_images/Car_Channel_0.png "Y Channel of Car"
[image4]: ./output_images/Car_Channel_1.png "Cr Channel of Car"
[image5]: ./output_images/Car_Channel_2.png "Cb Channel of Car"
[image6]: ./output_images/No_Car_Channel_0.png "Y Channel of No Car"
[image7]: ./output_images/No_Car_Channel_1.png "Cr Channel of No Car"
[image8]: ./output_images/No_Car_Channel_2.png "Cb Channel of No Car"
[image8]: ./output_images/ROI.png "Region of interest"
[image10]: ./output_images/FirstDetection.png "Initial detection"
[image11]: ./output_images/FinalDetection.png "Final detection"

### Pipeline

The resulting video can be found on YouTube by clicking on the image below:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/sWeFX5Ad_jM/0.jpg)](https://youtu.be/sWeFX5Ad_jM)

The executable code can be found in: `main.py`

### Detection

#### 1. Feature extraction and selection

The code for the feature extraction can be found within the file `feature.py`.  

First, all `vehicle` and `non-vehicle` images are read. For illustration an example of each class is visualized:

![alt text][image1]

![alt text][image2]

Then, different color spaces were explored by applying `skimage.hog()` which has the parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Various combinations of parameters and color spaces were tried but the final setup has been chosen as HOG features.
For the same example images as before, the `YCrCb` color space is used and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` are extraced. The results can be seen here:  

Y Channel of Car and its HOG features:  
![alt text][image3]

Cr Channel of Car and its HOG features:  
![alt text][image4]

Cb Channel of Car and its HOG features:  
![alt text][image5]

Y Channel of NO Car and its HOG features:  
![alt text][image6]

Cr Channel of NO Car and its HOG features:  
![alt text][image7]

Cb Channel of NO Car and its HOG features:  
![alt text][image8]

Moreover, the spatial feature with `spatial_size=(32,32)` and histogram features with `hist_bins=32` were incorporated in the final feature vector.

#### 3. Classifier training

A linear SVM with the above mentioned feature vector and as dataset 8500 examples of each class were used. Therefore, the data set was split up in 80% training examples and 20% test examples. As a result, a test accuracy of 99.7% was achieved

#### 4. Sliding window search

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image9]

#### 5. Detection resulits

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image10]

![alt text][image11]

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

