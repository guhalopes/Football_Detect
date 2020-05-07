import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing

frame = cv2.imread("BraBel.png")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

gray, thresh = PreProcessing.GrayThresh(frame)

x_ = []
y_ = []
w_ = []
h_ = []

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



for c in contours:
  x,y,w,h = cv2.boundingRect(c)
  x_.append(x)
  y_.append(y)
  w_.append(w)
  h_.append(h)
  

color = (255, 255, 0)
thickness = 3

for i in range(len(x_)):
    if(h_[i]>(1.2)*w_[i]):
        if(w_[i]>6 and h_[i]>6):
            player_image = frame[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
            player_hsv = cv2.cvtColor(player_image, cv2.COLOR_BGR2HSV)
            player_rgb = cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
            v0 = (x_[i], y_[i])
            vf = (x_[i]+w_[i], y_[i]+h_[i])
            cv2.rectangle(thresh, v0, vf, color, thickness)
            print('\nAverage color (BGR): ',np.array(cv2.mean(player_image[y:y+h,x:x+w])).astype(np.uint8))
            print('[x, y]: ', (x_[i], y_[i]))

           
"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        


# plot multiple images
plt.subplots(1, 2, figsize=(20, 15))


plt.subplot(1, 2, 1), plt.imshow(rgb, vmin = 0, vmax = 255)
plt.title('original')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(thresh, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('out')
plt.xticks([]),plt.yticks([])

plt.show()
plt.close()
