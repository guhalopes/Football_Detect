import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing
from WriteRec import Processing
from WriteCol import Colors

image = cv2.imread("BraBel.png")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rec,thresh = Processing.Rectangles(image)

rec_rgb = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)

# plot multiple images
plt.subplots(1, 2, figsize=(30, 30))


plt.subplot(1, 2, 1), plt.imshow(thresh, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('thresh')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(rec_rgb, vmin = 0, vmax = 255)
plt.title('players team identification')
plt.xticks([]),plt.yticks([])

plt.show()
plt.close()

