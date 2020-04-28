import cv2
import numpy as np
from matplotlib import pyplot as plt
from pre_proc import PreProcessing

img = cv2.imread('football.png')

gray, thresh = PreProcessing.preproc(img)
idx = 0
count = 0

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

# identify objects which height is greater then weight. those will be detect as players

for c in range(len(img)):
    x,y,w,h = cv2.boundingRect(c)
    
    # detect players
    if(h>=(1.5)*w):
        if(h>15, w>15):
            idx = idx + 1
            player_img = hsv[y:y+h, x:x+w]
            player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            
            # mask for france
            mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
            res1 = cv2.bitwise_and(player_img, player_img, mask1)
            res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
            nzCount = cv2.countNonZero(res1)
            
            # mask for belgium
            mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
            res2 = cv2.bitwise_and(player_img, player_img, mask2)
            res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
            res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
            nzCountred = cv2.countNonZero(res2)

if(nzCount >= 20):
     #Mark blue jersy players as france
     cv2.putText(image, 'France', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
     cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
else:
    pass


if(nzCountred>=20):
    #Mark red jersy players as belgium
    cv2.putText(image, 'Belgium', (x-2, y-2), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
else:
    pass          
            









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
