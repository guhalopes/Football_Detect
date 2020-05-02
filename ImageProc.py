import cv2
import numpy as np
from matplotlib import pyplot as plt
from FirstFilters import PreProcessing
from WriteRec import Processing
from WriteCol import Colors

image = cv2.imread("BraBel.png")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rec = Colors.ColorRec(image)


# plot multiple images
plt.subplots(2, 1, figsize=(20, 15))


plt.subplot(2, 1, 1), plt.imshow(rgb, vmin = 0, vmax = 255)
plt.title('original')
plt.xticks([]),plt.yticks([])

plt.subplot(2, 1, 2), plt.imshow(rec, vmin = 0, vmax = 255)
plt.title('players team identification')
plt.xticks([]),plt.yticks([])

plt.show()
plt.close()

