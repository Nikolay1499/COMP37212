import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint

#Function for performing a convolution using a 3X3 kernel
def convolution(image, weighted = False):

    #pad the image
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, [0, 0, 0])

    rows, cols = image.shape
    
    #defining the new image with sizes without the padding
    convImage = np.zeros((rows - 2, cols - 2))
    
    #the weighted parameters determines if the kernel will just take
    #the average of the values or will it weighted average
    if not weighted:
      kernel = np.ones((3,3), np.float32) / 9
    else:
      kernel = np.array([[0.8, 1.2, 0.8], [1.2, 2, 1.2], [0.8, 1.2, 0.8]]) / 10
      
    #loop over the image and compute the values of the new smoothed image
    for i in range(1, rows - 1):
      for j in range(1, cols - 1):
        for k in range(-1, 2):
          for l in range(-1, 2):
            convImage[i - 1][j - 1] += image[i + k][j + l] * kernel[k + 1][l + 1]
    
    convImage = convImage.astype(np.uint8)
    
    #display it on the screen and save the image
    cv2.imshow("Convoluted Kitty", convImage.astype(np.uint8))
    cv2.imwrite("smoothKitty.jpg", convImage)
    return convImage


def gradient(image, filename):
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, [0, 0, 0])
    
    rows, cols = image.shape
    
    tempGradientX = np.zeros((rows, cols))
    tempGradientY = np.zeros((rows, cols))
    gradientX = np.zeros((rows - 2, cols - 2))
    gradientY = np.zeros((rows - 2, cols - 2))
    gradientMagn = np.zeros((rows - 2, cols - 2))
    
    kernelX = np.array([-1, 0, 1])
    kernelY = np.array([1, 2, 1])
    
    for i in range(1, rows - 1):
      for j in range(1, cols - 1):
        for k in range(-1, 2):
          tempGradientX[i][j] += image[i][j + k] * kernelX[k + 1]
          tempGradientY[i][j] += image[i][j + k] * kernelY[k + 1]
          
    for i in range(1, rows - 1):
      for j in range(1, cols - 1):
        for k in range(-1, 2):
          gradientX[i - 1][j - 1] += tempGradientX[i + k][j] * kernelY[k + 1]
          gradientY[i - 1][j - 1] += tempGradientY[i + k][j] * kernelX[k + 1]
            
    gradientMagn = np.power(np.power(gradientX, 2.0) + np.power(gradientY, 2.0), 0.5)
    
    #hist_full = cv2.calcHist([gradientMagn.astype(np.uint8)],[0],None,[256],[0,256])
    #plt.plot(hist_full)
    #plt.xlim([0,256])
    #plt.show()
    cv2.imshow("Gradient X", gradientX.astype(np.uint8))
    cv2.imshow("Gradient Y", gradientY.astype(np.uint8))
    cv2.imshow("Gradient Magnitude", gradientMagn.astype(np.uint8))
        
    cv2.imwrite(filename, gradientMagn.astype(np.uint8))
    
  
img = cv2.imread("kitty.bmp", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("Kitty Original", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Convoluted Kitty", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gradient X", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gradient Y", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Gradient Magnitude", cv2.WINDOW_AUTOSIZE)  

while 1:
    
    cv2.imshow("Kitty Original", img)
    weightedImage = convolution(img, weighted = True)
    gradient(img, "kittyEdge.jpg")
    #gradient(weightedImage, "smoothKittyEdge.jpg")
    c = cv2.waitKey(500)
    if c == 27:
        break
        