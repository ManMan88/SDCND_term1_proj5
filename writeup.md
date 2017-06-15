## Vehicle Detection Project Writeup

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/not_car.png
[image3]: ./output_images/hog_source1.jpg
[image11]: ./output_images/hog1.jpg
[image4]: ./output_images/windows.jpg
[image5]: ./output_images/hot2.jpg
[image6]: ./output_images/hot5.jpg
[image7]: ./output_images/hot1.jpg
[image8]: ./output_images/heat.jpg
[image9]: ./output_images/labels.jpg
[image10]: ./output_images/final_img.jpg
[video1]: ./output_videos/project_video.mp4

## Rubric Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` and `single_img_features` functions in file `pipeline_functions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]  ![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:


![alt text][image3] ![alt text][image11]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and tested them by training a classifier and predicting test images. I finally chose the parameters that yeilded good results without too many features. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training the classifier I also used color features in the form hof spatial features (each image was resized to a 16x16 image) and histogram features (the histogram was divided to 64 bins). The code for this step is contained in the `bin_spatial` and `color_hist functions` in file `pipeline_functions.py`. 

I trained a non-linear SVM using rbf kernel, C parameter = 1, and automated gamma parameter. The data was normalized using `StandardScaler`. I used 80% of the data as training set and 20% as test set. The final classifier yielded an accuracy of 0.998 on the test set. The code for this step is contained in the file `classifier.py`. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the `slide_window` and `search_windows` functions in file `pipeline_functions.py`.
I decided on the window sizes and search locations by looking on the test images and project video. I saw what are the possible and reasonable positions and sizes of the cars, and chose the windows according to that. the window scales I chose are: 64x64, 96x96, 128x128, 160x160.
I then made the serach on the test images and video with different overlap values and finally setteled for overlap of 0.75 (75% overlap). 

Here is an example of a test image with all the search windows:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap. I used a buffer with a size of 5 (updating it each frame) and then thresholded the sum of the heatmaps to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image8]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image9]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image10]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced was that the pipeline took alot of time to process each image. I could have solved it by calculating the HOG features once for each image (more accurately, as the number of window scalses), and using the corresponding parameters for each search window. Furthermmore, I could have used a smarter search windows distribution which will use less windows, and possibly using adaptive windows distributions, such as once a car was found, then add more windows in its area.
In the final results, the found cars windows where a little wobbly and surrounded a bigger area than the car itself. I think the size of the window could be optimized by furthere search inside the found window, and in this way finding a more accurate window. Regarding the wobbliness, I could have improve the results by using the same algorithm I suggested, and also by creating a buffer that stores the final window size and applying some kind of filter on the window size (the same could be done with location) -> such as Kalman filter.

