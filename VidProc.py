import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing
from WriteRec import Processing
from WriteCol import Colors



vidcap = cv2.VideoCapture("BraBel.mp4")

if(vidcap.isOpened() == False):
    print('não tá rolando')

while(vidcap.isOpened()):
    ret, frame = vidcap.read()
    if(ret == True):
        
        rec = Colors.ColorRec(frame)
        
        cv2.imshow('Frame', rec)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    else:
        break
    
    
vidcap.release()

cv2.destroyAllWindows()
