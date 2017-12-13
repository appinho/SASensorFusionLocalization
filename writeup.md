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

A linear SVM with the above mentioned feature vector and as dataset 8500 examples of each class were used. Therefore, the data set was split up in 80% training examples and 20% test examples. As a result, a test accuracy of 99.7% was achieved. The code can be find in `classifier.py`.

#### 4. Sliding window search

A sliding window is applied with a window size of `xy_window = (96,96)` and an overlap of `xy_overlap = (0.5,0.5)` to search for detections of cars within the bottom part of the image. Therefore, a region of interest between the y-pixel values of `y_start_stop = [400,656]` is applied to save runtime and only look in potential regions. This region of interest is highlighted in the following image:  

![alt text][image9]

#### 5. Detection resulits

Ultimately, all detected parts of cars where gathered over the entire sliding window search. The result can be seen here.  

![alt text][image10]

With `scipy.ndimage.measurements import label` all overlapping windows are combined which leads to following final detection result.

![alt text][image11]

The code can be found within `heatmap.py`.

### Tracking

#### 1. Data association for removing False Positives



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

