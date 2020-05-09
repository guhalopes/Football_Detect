import cv2
import numpy as np
from matplotlib import pyplot as plt
from PreProcessing import PreProcessing
from DetectTeam import Colors


class Processing:
    
    def RectangleTeams(frame, lower, upper):
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gray, thresh = PreProcessing.GrayThresh(frame)
        
        """
        in: frame, rgb, gray, thresh
        parameters: lower_color, upper_color
        """
        
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
          
        """
        players -> frame[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
        """
          
        teams = []
        color = (255, 255, 255)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(len(x_)):
            if(h_[i]>(1.2)*w_[i]):
                if(w_[i]>6 and h_[i]>6):
                    player = frame[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
                    pre, out, rg, team = Colors.DetectTeams(player, lower_red, upper_red)
                    teams.append(team)
                    if(team == 1):
                        v0 = (x_[i], y_[i])
                        vf = (x_[i]+w_[i], y_[i]+h_[i])
                        final = cv2.rectangle(frame, v0, vf, color, thickness)
                        cv2.putText(frame, 'Inter', (x_[i]-2, y_[i]-2), font, 0.6, (0,0,255), 2, cv2.LINE_AA)
                    if(team == 0):
                        v0 = (x_[i], y_[i])
                        vf = (x_[i]+w_[i], y_[i]+h_[i])
                        final = cv2.rectangle(frame, v0, vf, color, thickness)
                        cv2.putText(frame, 'Gremio', (x_[i]-2, y_[i]-2), font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    
                
                    
        return teams, final

           
"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        
img = cv2.imread("GreNal.png")

lower_red = np.array([0,160,120])
upper_red = np.array([200,255,255])

# gremio -> use a range that goes to light blue til very dark
lower_blue = np.array([110,50,50])
upper_blue = np.array([145,255,255])

teams, out = Processing.RectangleTeams(img, lower_red, upper_red)

rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

img_in = cv2.imread("GreNal.png")
rgb_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

# plot multiple images
plt.subplots(1, 2, figsize=(20, 15))


plt.subplot(1, 2, 1), plt.imshow(rgb_in, vmin = 0, vmax = 255)
plt.title('original')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(rgb_out, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('out')
plt.xticks([]),plt.yticks([])

# plt.show()