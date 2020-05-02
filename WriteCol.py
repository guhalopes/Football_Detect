import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing

class Colors:
    
    def ColorRec(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         # yellow range
        lower_yellow = np.array([21, 180, 64])
        upper_yellow = np.array([40, 200, 255])
        
        # red range
        lower_red = np.array([0,160,50])
        upper_red = np.array([10,255,255])
        
        gray, thresh = PreProcessing.GrayThresh(image)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        colory = (0, 255, 255)
        colorr = (0,0,255)
        thickness = 2
        
        x_ = []
        y_ = []
        w_ = []
        h_ = []
        
        for c in contours:
          x,y,w,h = cv2.boundingRect(c)
          x_.append(x)
          y_.append(y)
          w_.append(w)
          h_.append(h)
          
        for i in range(len(x_)):
            if(h_[i]>(1.2)*w_[i]):
                if(w_[i]>6 and h_[i]>6):
                    player_image = image[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
                    player_hsv = cv2.cvtColor(player_image, cv2.COLOR_BGR2HSV)
                    player_rgb = cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
                    
                    mask1 = cv2.inRange(player_hsv, lower_yellow, upper_yellow)
                    res1_hsv = cv2.bitwise_and(player_image, player_image, mask = mask1)
                    res1_bgr = cv2.cvtColor(res1_hsv, cv2.COLOR_HSV2BGR)
                    res1_gray = cv2.cvtColor(res1_bgr, cv2.COLOR_BGR2GRAY)
                    nzCountY = cv2.countNonZero(res1_gray)
                    
                    mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                    res2_hsv = cv2.bitwise_and(player_image, player_image, mask = mask2)
                    res2_bgr = cv2.cvtColor(res2_hsv, cv2.COLOR_HSV2BGR)
                    res2_gray = cv2.cvtColor(res2_bgr, cv2.COLOR_BGR2GRAY)
                    nzCountR = cv2.countNonZero(res2_gray)
                    
                    if(nzCountY >= 15):
                        v0 = (x_[i], y_[i])
                        vf = (x_[i]+w_[i], y_[i]+h_[i])
                        cv2.rectangle(image, v0, vf, colory, thickness)
                        cv2.putText(image, 'Brazil', (x_[i]-2, y_[i]-2), font, 0.6,colory, 1, cv2.LINE_AA)
                    else:
                        pass
                    
                    if(nzCountR >= 20):
                        v0 = (x_[i], y_[i])
                        vf = (x_[i]+w_[i], y_[i]+h_[i])
                        cv2.rectangle(image, v0, vf, colorr, thickness)
                        cv2.putText(image, 'Belgium', (x_[i]-2, y_[i]-2), font, 0.6, colorr, 1, cv2.LINE_AA)
                    else:
                        pass
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            
            
            
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
# plt.imshow(image_rgb)
# plt.title('teams')
# plt.xticks([]),plt.yticks([])

# # plot multiple images
# plt.subplots(1, 2, figsize=(20, 15))


# plt.subplot(1, 2, 1), plt.imshow(player_rgb, vmin = 0, vmax = 255)
# plt.title('random player')
# plt.xticks([]),plt.yticks([])

# plt.subplot(1, 2, 2), plt.imshow(image_rgb, vmin = 0, vmax = 255)
# plt.title('players team identification')
# plt.xticks([]),plt.yticks([])

# plt.show()
# plt.close()

