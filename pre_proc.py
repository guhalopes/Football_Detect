"""
football field detection project

returns a grayscaled image in which the field is 0
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# read only one frame
img = cv2.imread('football.png')

# convert to hsv image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# green range
lower_green = np.array([40,40,40])
upper_green = np.array([70,255,255])
# blue range
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
# red range
lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])
# white range
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])

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




# # plot image
# plt.imshow(thresh, cmap = 'gray')
# plt.axis('off')
# plt.show()
# plt.close()

# plot multiple images
# plt.subplots(1, 3, figsize=(20, 15))

# plt.subplot(1, 3, 1), plt.imshow(img)
# plt.title('image')
# plt.xticks([]),plt.yticks([])

# plt.subplot(1, 3, 2), plt.imshow(res_gray, cmap='gray', vmin = 0, vmax = 255)
# plt.title('adj image')
# plt.xticks([]),plt.yticks([])

# plt.subplot(1, 3, 3), plt.imshow(thresh, cmap='gray', vmin = 0, vmax = 255)
# plt.title('gray image')
# plt.xticks([]),plt.yticks([])

# plt.show()
# plt.close()