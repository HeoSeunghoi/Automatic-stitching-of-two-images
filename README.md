# Automatic-stitching-of-two-images## Overview
This repository contains implementations of image warping techniques and feature matching using the ORB (Oriented FAST and Rotated BRIEF) algorithm, combined with RANSAC for robust estimation.

## ORB (Oriented FAST and Rotated BRIEF)
ORB is a fusion of the FAST keypoint detector and the BRIEF descriptor with many modifications to enhance performance. It provides rotation invariance and is patent-free, making it suitable for real-time applications and commercial use.

![Example Image for ORB](path_to_orb_image)

## RANSAC (Random Sample Consensus)
RANSAC is an iterative method used for robust parameter estimation. It's designed to handle a significant fraction of outliers and can be used with various model estimations.

![Example Image for RANSAC](path_to_ransac_image)   <!-- If you have a visual representation of RANSAC results, place the path here -->

## Input Images
Images used as input for the project:

![Input Image 1](images/image1.png)
![Input Image 2](images/image2.png)

## Results Images
Results after performing image warping and feature matching:

![Result Image](images/panorama_test.jpg)
