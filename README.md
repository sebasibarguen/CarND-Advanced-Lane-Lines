# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/original.jpg "Original"
[image2]: ./images/undistorted.png "Undistorted"
[image3]: ./images/transform.png "Road Transformed"
[image4]: ./images/gradient.png "Binary Example"
<!-- [image5]: ./images/warped.png "Warp Example" -->
[image6]: ./images/histogram_points.png "Window Points"
[image7]: ./images/color_fit_lines.png "Fit Visual"
[image8]: ./images/output.png "Output"
[hitogram]: ./images/histogram.png "Histogram"
[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This is an example of a raw image from the camera:

![alt text][image1]

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the the functions `calibrate_camera` and `cal_undistort(img, objpoints, imgpoints)` inside `find_lanes.py` file.

Using the chessboard images in `/camera_cal` I preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are inside the function `pipeline` inside `find_lanes.py`).  Here's an example of my output for this step.

![alt text][image4]

To get to this image, I converted the image from RGB to HLS format. I focused on the S channel and applied a sobel gradient. Then I combined a threshold image of the gradient with a threshold colored image.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image(img)`.  This function takes as inputs an image (`img`), and transforms the image using the below *source* and *destination* points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.array([[590. /1280.*img_size[1], 455./720.*img_size[0]],
                [690. /1280.*img_size[1], 455./720.*img_size[0]],
                [1090./1280.*img_size[1], 720./720.*img_size[0]],
                [190. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)

dst = np.array([[300. /1280.*img_size[1], 100./720.*img_size[0]],
                [1000./1280.*img_size[1], 100./720.*img_size[0]],
                [1000./1280.*img_size[1], 720./720.*img_size[0]],
                [300. /1280.*img_size[1], 720./720.*img_size[0]]], np.float32)

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 590, 455      | 300, 100      |
| 690, 455      | 1000, 100     |
| 1090, 720     | 1000, 720     |
| 190, 720      | 300, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I plotted a histogram of the wrapped image to see where the most likely location was for the lane lines. This is the histogram:

![alt text][histogram]

Using the histogram, I can then use it to plot points throught the `y` position or height of the image to map the lane lines. I did this by breaking down the image into sections or windows, and searching the two peaks in the histogram for each image window. After plotting these points, I got the following mapping:

![alt text][image6]

With these points, then it's straightforward to calculate a polynomial line that best fits the points. This is done in the function `poly_fit` inside `find_lanes.py`.

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature of the lanes was calculated based on the previously found points. Using the provided mapping of pixels per meter of 30/720 mts/pixels in the y dimension and 3.7/720 mts/pixels in the x dimension, I calculated the curvature using the function
```
# Not real code. Pseudocode
curve = (1 + (2*A + B)**2)**(3/2) / |2*A|
```

This is also calculated inside the function `poly_fit` in `find_lanes.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The video processing is based on the single image pipeline. The only difference is that it applies a moving average to the position of the fitted lines. I found that this helps make the video smoother and reduce the amount of noise between frames.
