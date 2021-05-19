import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
        
def RatioFeatureMatcher(desc1, desc2):
    matches = []
    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # TODO 8: Perform ratio feature matching.
    # This uses the ratio of the SSD distance of the two best matches
    # and matches a feature in the first image with the closest feature in the
    # second image.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # You don't need to threshold matches in this function
    # TODO-BLOCK-BEGIN
    num_features, num_feature_dims = desc1.shape
    dists = spatial.distance.cdist(desc1, desc2)
    mins = np.argmin(dists, axis=1)
    for i in range(len(mins)):
        distsi = dists[i]
        bestMatch = mins[i]
        bestDist =  np.amin(distsi)
        #alter distsi to change the previous min to a really big number
        distsi[mins[i]] = float('inf')
        secondBestMatch = np.argmin(distsi)
        secondBestDist =  np.amin(distsi)
        
        ratioDist = bestDist/float(secondBestDist)
                    
        m = cv2.DMatch(_queryIdx = i, _trainIdx = mins[i], _distance = ratioDist)
        matches.append(m)
    
    # TODO-BLOCK-END

    return matches
    
def SSDFeatureMatcher(desc1, desc2):
    matches = []
    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # TODO 7: Perform simple feature matching.  This uses the SSD
    # distance between two feature vectors, and matches a feature in
    # the first image with the closest feature in the second image.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # TODO-BLOCK-BEGIN
    num_features, num_feature_dims = desc1.shape
    dists = spatial.distance.cdist(desc1, desc2)
    mins = np.argmin(dists, axis=1)
    for i in range(len(mins)):
        m = cv2.DMatch(_queryIdx = i, _trainIdx = mins[i], _distance = np.amin(dists[i]))
        matches.append(m)
    # TODO-BLOCK-END

    return matches
        
def HarrisPointsDetector(image, thresholdValue = -3.0):
    image = image.astype(np.float32)
    image /= 255.
    height, width = image.shape[:2]
    features = []

    # Create grayscale image used for Harris detection
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # computeHarrisValues() computes the harris score at each pixel
    # position, storing the result in harrisImage.
    # You will need to implement this function.
    height, width = grayImage.shape[:2]

    harrisImage = np.zeros(grayImage.shape[:2])
    orientationImage = np.zeros(grayImage.shape[:2])

    # TODO 1: Compute the harris corner strength for 'srcImage' at
    # each pixel and store in 'harrisImage'.  See the project page
    # for direction on how to do this. Also compute an orientation
    # for each pixel and store it in 'orientationImage.'
    # TODO-BLOCK-BEGIN
    xdir = scipy.ndimage.sobel(grayImage, axis=1, mode="reflect")
    xsq = np.square(xdir)

    ydir = scipy.ndimage.sobel(grayImage, axis=0, mode="reflect")
    ysq = np.square(ydir)
    xTy = np.multiply(xdir, ydir)
    
    xsq_masked = ndimage.filters.gaussian_filter(xsq, .5)
    ysq_masked = ndimage.filters.gaussian_filter(ysq, .5)
    xTy_masked = ndimage.filters.gaussian_filter(xTy, .5)

    det_mat = np.multiply(xsq_masked, ysq_masked) - np.multiply(xTy_masked, xTy_masked)
    trace_mat = xsq_masked + ysq_masked
    trace_mat_sq_T_point1 = np.square(trace_mat)*.1
    harrisImage = det_mat - trace_mat_sq_T_point1

    orientationImage = np.rad2deg(np.arctan2(ydir,xdir))

    # Compute local maxima in the Harris image.  You will need to
    # implement this function. Create image to store local maximum harris
    # values as True, other pixels False
    harrisMaxImage = np.zeros_like(harrisImage, np.bool)

    # TODO 2: Compute the local maxima image
    maxes = ndimage.filters.maximum_filter(harrisImage,7,mode='constant', cval = -1e100)
    # TODO-BLOCK-BEGIN
    
    harrisMaxImage = maxes == harrisImage

    # Loop through feature points in harrisMaxImage and fill in information
    # needed for descriptor computation for each point.
    # You need to fill x, y, and angle.
    for y in range(height):
        for x in range(width):
            if not harrisMaxImage[y, x]:
                continue

            f = cv2.KeyPoint()

            # TODO 3: Fill in feature f with location and orientation
            # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
            # f.angle to the orientation in degrees and f.response to
            # the Harris score
            # TODO-BLOCK-BEGIN
            f.pt = (x, y)
            # Dummy size
            f.size = 10
            f.angle = orientationImage[y][x]
            f.response = harrisImage[y][x]
            # TODO-BLOCK-END

            features.append(f)
    threshold = 10**float(thresholdValue)
    if features is not None:
        kps = [kp for kp in features if kp.response >= threshold]
        
    return kps    
    
orb = cv2.ORB_create()

#For filtering the matches, 100 for all matches, 0 for none
percent = 10


#Create windows for displaying images
cv2.namedWindow("Harris First Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Harris Second Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Matches", cv2.WINDOW_AUTOSIZE)

#Open first images, indentify interest points, use ORB feature descriptor 
image = cv2.imread("mYWhGE8Q.jpeg", cv2.IMREAD_COLOR)
kps = HarrisPointsDetector(image)
harrisImage = cv2.drawKeypoints(image, kps, None, color=(0, 255, 0))
keypoints, desc = orb.compute(image, kps)

#Open second images, indentify interest points, use ORB feature descriptor 
image2 = cv2.imread("brighterBernie.jpg", cv2.IMREAD_COLOR)
kps2 = HarrisPointsDetector(image2)
harrisImage2 = cv2.drawKeypoints(image2, kps2, None, color=(0, 255, 0))
keypoints2, desc2 = orb.compute(image2, kps2)

#Match the features using SSD or Ratio
matches = sorted(SSDFeatureMatcher(desc,desc2), key = lambda x : x.distance)

#create the matched image
matchImage = cv2.drawMatches(harrisImage, kps, harrisImage2, kps2, matches[::int(100 / percent)] if percent > 0 else [], None,flags=2)

#Show images
cv2.imshow("Harris First Image", cv2.resize(harrisImage.astype(np.uint8), (700, 700)))
cv2.imshow("Harris Second Image", cv2.resize(harrisImage2.astype(np.uint8), (700, 700)))
cv2.imshow("Matches", cv2.resize(matchImage.astype(np.uint8), (1400, 700)))
cv2.waitKey(0)