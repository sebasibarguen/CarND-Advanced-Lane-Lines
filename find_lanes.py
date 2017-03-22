import numpy as np
import cv2
import glob
from collections import deque
import matplotlib.pyplot as plt

import os
import random
import math

import scipy
from scipy import signal


def calibrate_camera(calib_img_dir='./camera_cal/calibration*.jpg'):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calib_img_dir)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def region_of_interest(img):

    shape = img.shape
    vertices = np.array([[(0,0),(shape[1],0),(shape[1],0),(6*shape[1]/7,shape[0]),
                      (shape[1]/7,shape[0]), (0,0)]],dtype=np.int32)

    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def undistort(img):
    result = cv2.undistort(img, mtx, dist, None, mtx)
    return result

def binarize(img, s_thresh=(120, 255), sx_thresh=(20, 255), l_thresh=(40,255)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    # sobelx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255))
    # l_channel_col=np.dstack((l_channel,l_channel, l_channel))
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    channels = 255*np.dstack(( l_binary, sxbinary, s_binary)).astype('uint8')
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
    binary = 255*np.dstack((binary, binary, binary)).astype('uint8')
    return  binary, channels

def warp(img, tobird=True):
    corners = np.float32([[190,720], [589,457], [698,457], [1145,720]])
    new_top_left = np.array([corners[0,0],0])
    new_top_right = np.array([corners[3,0],0])
    offset = [150,0]

    img_size = (img.shape[1], img.shape[0])
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([corners[0]+offset, new_top_left+offset, new_top_right-offset, corners[3]-offset])
    if tobird:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, img_size , flags=cv2.INTER_LINEAR)
    return warped, M


def find_peaks(img, thresh):
    img_half = img[int(img.shape[0]/2):,:,0]
    data = np.sum(img_half, axis=0)
    filtered = scipy.ndimage.filters.gaussian_filter1d(data, 20)
    xs = np.arange(len(filtered))
    peak_ind = signal.find_peaks_cwt(filtered, np.arange(20, 300))
    peaks = np.array(peak_ind)
    peaks = peaks[filtered[peak_ind] > thresh]
    return peaks, filtered


def get_next_window(img, center_point, width):

    ny,nx,_ = img.shape
    mask = np.zeros_like(img)
    if (center_point <= width/2): center_point = width/2
    if (center_point >= nx-width/2): center_point = nx-width/2

    left  = center_point - width/2
    right = center_point + width/2

    vertices = np.array([[(left, 0),(left, ny), (right, ny),(right, 0)]], dtype=np.int32)
    ignore_mask_color = (255, 255, 255)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(mask,img)

    hist = np.sum(masked[:,:,0],axis=0)
    if max(hist>10000):
        center = np.argmax(hist)
    else:
        center = center_point

    return masked, center

def lane_from_window(binary, center_point, width):
    n_zones = 6
    ny,nx,nc = binary.shape
    zones = binary.reshape(n_zones,-1,nx,nc)
    zones = zones[::-1] # start from the bottom slice
    window,center = get_next_window(zones[0], center_point, width)

    for zone in zones[1:]:
        next_window,center = get_next_window(zone, center, width)
        window = np.vstack((next_window,window))

    return window


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, n=5):
        # length of queue to store data
        self.n = n
        #number of fits in buffer
        self.n_buffered = 0
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque([],maxlen=n)
        #average x values of the fitted line over the last n iterations
        self.avgx = None
        # fit coeffs of the last n fits
        self.recent_fit_coeffs = deque([],maxlen=n)
        #polynomial coefficients averaged over the last n iterations
        self.avg_fit_coeffs = None
        # xvals of the most recent fit
        self.current_fit_xvals = [np.array([False])]
        #polynomial coefficients for the most recent fit
        self.current_fit_coeffs = [np.array([False])]
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #y values for line fit
        self.fit_yvals = np.linspace(0, 100, num=101)*7.2  # always the same y-range as image
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        # origin (pixels) of fitted line at the bottom of the image
        self.line_pos = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

    def add_values(self):
        self.recent_xfitted.appendleft(self.current_fit_xvals)
        self.recent_fit_coeffs.appendleft(self.current_fit_coeffs)
        assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
        self.n_buffered = len(self.recent_xfitted)

    def clear_buffer(self):
        if self.n_buffered > 0:
            self.recent_xfitted.pop()
            self.recent_fit_coeffs.pop()
            assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
            self.n_buffered = len(self.recent_xfitted)

        return self.n_buffered

    def set_avgx(self):
        fits = self.recent_xfitted
        if len(fits)>0:
            avg=0
            for fit in fits:
                avg +=np.array(fit)
            avg = avg / len(fits)
            self.avgx = avg

    def set_avgcoeffs(self):
        coeffs = self.recent_fit_coeffs
        if len(coeffs)>0:
            avg=0
            for coeff in coeffs:
                avg +=np.array(coeff)
            avg = avg / len(coeffs)
            self.avg_fit_coeffs = avg


    def set_line_base_pos(self):
        y_eval = max(self.fit_yvals)
        self.line_pos = self.current_fit_coeffs[0]*y_eval**2 \
                        +self.current_fit_coeffs[1]*y_eval \
                        + self.current_fit_coeffs[2]
        basepos = 640

        self.line_base_pos = (self.line_pos - basepos)*3.7/600.0 # 3.7 meters is about 600 pixels in the x direction

    # here come sanity checks of the computed metrics
    def accept_lane(self):
        flag = True
        maxdist = 2.8  # distance in meters from the lane
        if(abs(self.line_base_pos) > maxdist ):
            print('lane too far away')
            flag  = False
        if(self.n_buffered > 0):
            relative_delta = self.diffs / self.avg_fit_coeffs
            # allow maximally this percentage of variation in the fit coefficients from frame to frame
            if not (abs(relative_delta)<np.array([0.7,0.5,0.15])).all():
                print('fit coeffs too far off [%]',relative_delta)
                flag=False

        return flag

    def update(self, lane):

        self.ally, self.allx = (lane[:,:,0]>254).nonzero()

        self.current_fit_coeffs = np.polyfit(self.ally, self.allx, 2)
        yvals = self.fit_yvals
        self.current_fit_xvals = self.current_fit_coeffs[0]*yvals**2 + self.current_fit_coeffs[1]*yvals + self.current_fit_coeffs[2]

        y_eval = max(self.fit_yvals)
        if self.avg_fit_coeffs is not None:
            self.radius_of_curvature = ((1 + (2*self.avg_fit_coeffs[0]*y_eval + self.avg_fit_coeffs[1])**2)**1.5) \
                             /np.absolute(2*self.avg_fit_coeffs[0])

        y_eval = max(self.fit_yvals)
        self.line_pos = self.current_fit_coeffs[0]*y_eval**2 \
                        +self.current_fit_coeffs[1]*y_eval \
                        + self.current_fit_coeffs[2]
        basepos = 640

        self.line_base_pos = (self.line_pos - basepos)*3.7/600.0 # 3.7 meters is about 600 pixels in the x direction


        if self.n_buffered > 0:
            self.diffs = self.current_fit_coeffs - self.avg_fit_coeffs
        else:
            self.diffs = np.array([0,0,0], dtype='float')

        if self.accept_lane():
            self.detected = True
            self.add_values()
            self.set_avgx()
            self.set_avgcoeffs()
        else:
            self.detected = False
            self.clear_buffer()
            if self.n_buffered > 0:
                self.set_avgx()
                self.set_avgcoeffs()

        return self.detected, self.n_buffered

#######################################################
def get_binary_lane_image(img, line, window_center, width=300):
    if line.detected:
        window_center = line.line_pos
    else:
        peaks, filtered = find_peaks(img, thresh=3000)

        peak_ind = np.argmin(abs(peaks - window_center))
        peak  = peaks[peak_ind]
        window_center = peak

    lane_binary = lane_from_window(img, window_center, width)
    return lane_binary


def project_lane_lines(img, left_fitx, right_fitx, yvals):

    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    undist = cal_undistort(img, objpoints, imgpoints)
    unwarp,Minv = warp(img,tobird=False)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


class FindLanes:

    def __init__(self,):

        self.left = Line(7)
        self.right = Line(7)

        objpoints, imgpoints = calibrate_camera()
        self.objpoints = objpoints
        self.imgpoints = imgpoints

    def process_image(self, img):

        undist = cal_undistort(img, self.objpoints, self.imgpoints)
        binary,_  = binarize(undist)
        warped,_  = warp(binary)
        warped_binary = region_of_interest(warped)

        window_center_l = 340
        if self.left.detected:
            window_center_l = self.left.line_pos
        left_binary = get_binary_lane_image(warped_binary, self.left, window_center_l, width=300)

        window_center_r = 940
        if self.right.detected:
            window_center_r = self.right.line_pos
        right_binary = get_binary_lane_image(warped_binary, self.right, window_center_r, width=300)

        detected_l, n_buffered_left = self.left.update(left_binary)
        detected_r, n_buffered_right = self.right.update(right_binary)

        left_fitx = self.left.avgx
        right_fitx = self.right.avgx
        yvals = self.left.fit_yvals
        lane_width = 3.7
        off_center = -100*round(0.5*(self.right.line_base_pos-lane_width/2) +  0.5*(abs(self.left.line_base_pos)-lane_width/2),2)

        result = project_lane_lines(img, left_fitx, right_fitx, yvals)

        return result

find_lanes = FindLanes()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_video", help="video to process", type=str)
args = parser.parse_args()

video_file = args.input_video
output_file = 'output_' + video_file


print('Processing video ...', video_file)
from moviepy.editor import VideoFileClip

clip2 = VideoFileClip(video_file)
vid_clip = clip2.fl_image(find_lanes.process_image)
vid_clip.write_videofile(output_file, audio=False)
