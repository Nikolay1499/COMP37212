import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint

#Function for performing a convolution using a 3X3 kernel
def convolution(image, weighted = False):

    #pad the image
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType, None, value)

    rows, cols = image.shape
    
    #defining the new image with sizes without the padding
    convImage = np.zeros((rows - 2, cols - 2))
    
    #the weighted parameters determines if the kernel will just take
    #the average of the values or will it weighted average
    if not weighted:
      kernel = np.ones((3,3), np.float32) / 9
    else:
      kernel = np.array([[0.5, 1, 0.5], [1, 2, 1], [0.5, 1, 0.5]]) / 8
      
    #loop over the image and compute the values of the new smoothed image
    for i in range(1, rows - 1):
      for j in range(1, cols - 1):
        for k in range(-1, 2):
          for l in range(-1, 2):
            convImage[i - 1][j - 1] += image[i + k][j + l] * kernel[k + 1][l + 1]
        
    #display it on the screen
    cv2.imshow("Convoluted Kitty", convImage.astype(np.uint8))

img = cv2.imread("kitty.bmp", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("Kitty Original", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Convoluted Kitty", cv2.WINDOW_AUTOSIZE)
  
top = int(0.05 * img.shape[0])  # shape[0] = rows
bottom = top
left = int(0.05 * img.shape[1])  # shape[1] = cols
right = left
borderType = cv2.BORDER_CONSTANT
while 1:
      
    value = [randint(0, 255), randint(0, 255), randint(0, 255)]
    value = [0, 0, 0]
    dst = cv2.copyMakeBorder(img, 1, 1, 1, 1, borderType, None, value)
    
    cv2.imshow("Kitty Original", img)
    convolution(img, weighted = True)
    c = cv2.waitKey(500)
    if c == 27:
        break
    elif c == 99: # 99 = ord('c')
        borderType = cv2.BORDER_CONSTANT
    elif c == 114: # 114 = ord('r')
        borderType = cv2.BORDER_REPLICATE
        