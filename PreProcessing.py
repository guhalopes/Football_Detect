import cv2
import numpy as np
from matplotlib import pyplot as plt


class PreProcessing:
    
    def __init__(self, img):
        self.img = img 
        
    
    def GrayThresh(img):
        
        # convert to hsv image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # green range
        lower_green = np.array([40,40,40])
        upper_green = np.array([70,255,255])
              
        # define a mask with upper and lower values
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # apply mask
        res = cv2.bitwise_and(img, img, mask=mask)

        # hsv to gray
        res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)
        
        # define some morphological operations to reduce noise
        kernel = np.ones((13,13), np.uint8)
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV |  cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return res_gray, thresh


frame = cv2.imread("BraBel.png")
gray, thresh = PreProcessing.GrayThresh(frame)

# plot multiple images
plt.subplots(1, 2, figsize=(20, 15))


plt.subplot(1, 2, 1), plt.imshow(gray, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('original')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(thresh, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('out')
plt.xticks([]),plt.yticks([])

plt.show()
plt.close()
