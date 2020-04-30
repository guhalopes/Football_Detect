import cv2
import numpy as np
from matplotlib import pyplot as plt
from pre_proc import PreProcessing

image = cv2.imread('BraBel.png')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray, thresh = PreProcessing.preproc(image)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
idx = 0
for i in contours:
    x, y, w, h = cv2.boundingRect(i)
    
    if(h>(1,3)*w):
        if(w>3 and h>3):
            idx = idx + 1
            player_image = image[x:x+w, y:y+h]
            player_hsv = cv2.cvtColor(player_image, cv2.COLOR_BGR2HSV)
            
            
            
    


# plot multiple images
plt.subplots(1, 2, figsize=(20, 15))


plt.subplot(1, 2, 1), plt.imshow(gray, cmap='gray', vmin = 0, vmax = 255)
plt.title('gray image')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(thresh, cmap='gray', vmin = 0, vmax = 255)
plt.title('threshold image')
plt.xticks([]),plt.yticks([])

plt.show()
plt.close()

