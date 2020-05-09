import cv2
import numpy as np
from matplotlib import pyplot as plt
from PreProcessing import PreProcessing

class Colors:
    
    def ColorRec(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # teamA = input("\ndefine team A\n")
        # teamB = input("\ndefine team B\n")
        
        # colorA = input("\ndefine color for team A\n")
        # colorB = input("\ndefine color for team B\n")
        
        

         # yellow range
        lower_yellow = np.array([21, 180, 64])
        upper_yellow = np.array([40, 200, 255])
        
        # red range
        lower_red = np.array([0,160,50])
        upper_red = np.array([10,255,255])
        
        gray, thresh = PreProcessing.GrayThresh(img)
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
        player_ = []
        
        for c in contours:
          x,y,w,h = cv2.boundingRect(c)
          x_.append(x)
          y_.append(y)
          w_.append(w)
          h_.append(h)
        
        for i in range(len(x_)):
            if(h_[i]>(1.2)*w_[i]):
                if(w_[i]>6 and h_[i]>6):
                    player_image = img[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
                    player_hsv = cv2.cvtColor(player_image, cv2.COLOR_BGR2HSV)
                    player_rgb = cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
                    plt.imshow(player_rgb)
                    plt.xticks([]),plt.yticks([])
                    mask1 = cv2.inRange(player_hsv, lower_yellow, upper_yellow)
                    res1_hsv = cv2.bitwise_and(player_image, player_image, mask = mask1)
                    res1_bgr = cv2.cvtColor(res1_hsv, cv2.COLOR_HSV2BGR)
                    res1_gray = cv2.cvtColor(res1_bgr, cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1_gray)
                    
                    mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                    res2_hsv = cv2.bitwise_and(player_image, player_image, mask = mask2)
                    res2_bgr = cv2.cvtColor(res2_hsv, cv2.COLOR_HSV2BGR)
                    res2_gray = cv2.cvtColor(res2_bgr, cv2.COLOR_BGR2GRAY)
                    nzCountR = cv2.countNonZero(res2_gray)
                    
        if(nzCount >= 15):
            v0 = (x_[i], y_[i])
            vf = (x_[i]+w_[i], y_[i]+h_[i])
            image = cv2.rectangle(img, v0, vf, colory, thickness)
            cv2.putText(image, 'Brazil', (x_[i]-2, y_[i]-2), font, 0.6,colory, 1, cv2.LINE_AA)
        else:
            pass
        
        if(nzCountR >= 20):
            v0 = (x_[i], y_[i])
            vf = (x_[i]+w_[i], y_[i]+h_[i])
            image = cv2.rectangle(img, v0, vf, colorr, thickness)
            cv2.putText(image, 'Belgium', (x_[i]-2, y_[i]-2), font, 0.6, colorr, 1, cv2.LINE_AA)
        else:
            pass
                
      
        return image
        
            
"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        

# img = cv2.imread("BraBel.png")
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# out = Colors.ColorRec(img)


# # plot multiple images
# plt.subplots(1, 2, figsize=(20, 15))


# plt.subplot(1, 2, 1), plt.imshow(img, vmin = 0, vmax = 255)
# plt.title('original')
# plt.xticks([]),plt.yticks([])

# plt.subplot(1, 2, 2), plt.imshow(out, vmin = 0, vmax = 255)
# plt.title('out')
# plt.xticks([]),plt.yticks([])

# plt.show()
# plt.close()

