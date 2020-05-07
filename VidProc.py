import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing
from WriteRec import Processing
from WriteCol import Colors

# Capture video file
vidcap = cv2.VideoCapture("cutvideo.mp4")

# Check if the system can open the video file
if(vidcap.isOpened() == False):
    print('não tá rolando')
    
# Default resolutions of the frame are obtained. That depends on the system
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))

# Define the codec and create VideoWriter object
# The output is stored in 'outpy' file
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
# Cria o loop para os frames
while(vidcap.isOpened()):
    ret, frame = vidcap.read()
    if(ret == True):
                
        # Define a função que vai ser usada nos frames
        gray =cv2.blur(frame,(5,5))
        
        # Escreve os frames para a saída
        out.write(gray)

        # # Display saída
        # cv2.imshow('Frame', rec)
        
        # 'q' acaba com o loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    else:
        break
    
    
vidcap.release()
out.release()

cv2.destroyAllWindows()
