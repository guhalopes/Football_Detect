import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing


class Processing:
    
    def Rectangles(image):
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gray, thresh = PreProcessing.GrayThresh(image)
        
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
          
        
        color = (255, 255, 255)
        thickness = 2
        
        for i in range(len(x_)):
            if(h_[i]>(1.2)*w_[i]):
                if(w_[i]>6 and h_[i]>6):
                    v0 = (x_[i], y_[i])
                    vf = (x_[i]+w_[i], y_[i]+h_[i])
                    final = cv2.rectangle(rgb, v0, vf, color, thickness)
                    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return final
