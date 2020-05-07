import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing
from WriteRec import Processing

count = 0

vidcap = cv2.VideoCapture('cutvideo.mp4')
success,image = vidcap.read()

success = True

while success:
	#run function to detect players
	rectangles = Processing.Rectangles(image)
    
    
    
	print('Read a new frame: ', success)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	success,image = vidcap.read()
    
	print('Read a new frame: ', success)     # save frame as JPEG file	
	count += 1
	cv2.imshow('Match Detection', rectangles)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	success,image = vidcap.read()
    
vidcap.release()
cv2.destroyAllWindows()