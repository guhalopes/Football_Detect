import cv2
import numpy as np
from matplotlib import pyplot as plt
from PreProcessing import PreProcessing


class Processing:
    
    def Rectangles(frame):
        
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
          
        player_ = []
        color = (255, 255, 255)
        thickness = 2
        
        for i in range(len(x_)):
            if(h_[i]>(1.2)*w_[i]):
                if(w_[i]>6 and h_[i]>6):
                    player_image = frame[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
                    player_rgb = cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
                    player_.append(player_image)
                    v0 = (x_[i], y_[i])
                    vf = (x_[i]+w_[i], y_[i]+h_[i])
                    final = cv2.rectangle(rgb, v0, vf, color, thickness)
                    
        output = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                    
        return output,player_

           
"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        

# image = cv2.imread("GreNal.png")
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# out, players = Processing.Rectangles(image)
# out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# # plot multiple images
# plt.subplots(1, 2, figsize=(20, 15))


# plt.subplot(1, 2, 1), plt.imshow(players[2], vmin = 0, vmax = 255)
# plt.title('original')
# plt.xticks([]),plt.yticks([])

# plt.subplot(1, 2, 2), plt.imshow(out, vmin = 0, vmax = 255)
# plt.title('out')
# plt.xticks([]),plt.yticks([])

# plt.show()
# plt.close()

# cv2.imwrite("player_test.png", players[2])