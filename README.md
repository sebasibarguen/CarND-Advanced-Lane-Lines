# Advanced Lane Finding Project

The goals of this project is build an algorithm that detects lane lines and displays then in the image. Te following steps were followed:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms and gradients to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/original.jpg "Original"
[image2]: ./images/undistorted.png "Undistorted"
[image3]: ./images/transform.png "Road Transformed"
[image4]: ./images/sobel.png "Binary Example"
[image5]: ./images/warped.png "Warp Example"
[image6]: ./images/windows.png "Window Points"
[image7]: ./images/polynomial.png "Polynomial Fit"
[image8]: ./images/result.png "Output"
[hitogram]: ./images/histogram.png "Histogram"
[video1]: ./output_project_video.mp4 "Video"

## Description

The projects purpose is to find lane lines from images coming from a front facing camera in a car. This is an example of a raw image from the camera:

![alt text][image1]

With just the pixels from the camera, the algorithm is able to detect both **right** and **left** lanes. The pipeline follows a traditional computer vision approach.

### Camera Calibration

Using the chessboard images in `/camera_cal` I preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image2]

> Reference: `Find Lane Lines.ipynb`, code block `20`, `calibrate_camera` function.

### Distortion Correction

After calibrating the images, we can now correclty distort the camera images so to remove any noise from the camera. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

> Reference: `Find Lane Lines.ipynb`, code block `20`, `undistort` function.

#### Binary Images
Next, I used a combination of color and gradient thresholds to generate a binary image. Here's an example of a test image when converted to binary:

![alt text][image4]

To get to this image, I converted the image from RGB to HLS format. I focused on the S channel and applied a sobel gradient. Then I combined a threshold image of the gradient with a threshold colored image.

> Reference: `Find Lane Lines.ipynb`, code block `20`, `binarize` function.

#### Transformed

To find the lanes, I transformed the image to reduce the search space inside the image and make it easier to later use an algorithmic approach of finding lanes.

The code for my perspective transform includes a function called `warp`.  This function takes as inputs an image (`img`), and transforms the image. The transformation


This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 590, 455      | 300, 100      |
| 690, 455      | 1000, 100     |
| 1090, 720     | 1000, 720     |
| 190, 720      | 300, 720      |

I verified that my perspective transform was working as expected by verifying that in the warped images, the lane lines appeared aproximatley parallel and vertical.

![alt text][image3]

> Reference: `Find Lane Lines.ipynb`, code block `20`, `warp` function.

#### Lane Line Pixels and Polyfit

To find the lane lines, I used a window search algorithm to plot a histogram of most-likely location of the lane line in the given window. The histogram plotted the values for the pixels, here is an example:

![alt text][histogram]

Using the histogram, I can then use it to plot points throught the `y` position or height of the image to map the lane lines. I did this by breaking down the image into sections or windows, and searching the two peaks in the histogram for each image window. After plotting these points, I got the following mapping:

![alt text][image6]

With these points, then it's straightforward to calculate a polynomial line that best fits the points. This is done using the numpy function `np.polyfit`

![alt text][image7]

> Reference: `Find Lane Lines.ipynb`, code block `20`, inside `Lane` class, `update` method.

#### Curvature

The curvature of the lanes was calculated based on the previously found points. Using the provided mapping of pixels per meter of 30/720 mts/pixels in the y dimension and 3.7/720 mts/pixels in the x dimension, I calculated the curvature using the function

```
# Not real code. Pseudocode
curve = (1 + (2*A + B)**2)**(3/2) / |2*A|
```

The curvature normaly lie between 300 mts and 3000 mts. If the curvature was way off (in the case of the video), the results from the pipeline were rejected for that frame, and I used the previous values as reference.

> Reference: `Find Lane Lines.ipynb`, code block `20`, inside `Lane` class, `update` method.

#### Result

The end result of the whole process can be seen in the following image:

![alt text][image8]

The lane lines have been marked with yellow, and the lane was filled with green.

---

### Pipeline (video)

#### Video Result

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My initial take on the project was using the same process for a single image, and apply it to the video. This had many difficulties, given that many images the lane lines were not being recognized with a high degree of confidence. This is were having sanity checks, and validating that the results from the lane finding pipeline make sense has big benefits. I also saw huge improvements in performance by averaging frames and including a buffer.

After implementing the `Line` and `FindLane` classes, it made the video result more smooth.
