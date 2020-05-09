import cv2
import numpy as np
from matplotlib import pyplot as plt
from PreProcessing import PreProcessing

""""
new approach:
    use the player tracking function and use only one color
    (everything that is red belongs to team A. the rest is team B)
    our priority should be plained shirts
"""

def count_noncolor_np(img):
    """Return the number of pixels in img that are not colorA.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()

#input colors should be np.array
def DetectTeams(image, lower, upper):

    
    # convert to hsv
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    # mask the field in order to improve the rate
    # green range
    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])
    
    # define a mask with upper and lower values
    mask = cv2.inRange(image_hsv, lower_green, upper_green)
    image_nofield = cv2.bitwise_not(image, image, mask=mask)
    image_nofield_hsv = cv2.cvtColor(image_nofield, cv2.COLOR_BGR2HSV)
    image_nofield = cv2.cvtColor(image_nofield, cv2.COLOR_BGR2RGB)
    
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image_nofield_hsv, lower, upper)
    output = cv2.bitwise_and(image_nofield, image_nofield, mask = mask)
    tot_pix = count_noncolor_np(image_nofield)
    color_pix = count_noncolor_np(output)
    ratio = color_pix/tot_pix
    
    ## Find the team
    print(ratio)
    return ratio, image_nofield, output
    # if max_ratio < 0.20:
    #     print('not sure')
    #     return 'not_sure'
    # else:
    #     ## Identify black pixels first and return them first
    #     if ratioList[1] >= 0.30:
    #         print("black")
    #         return 'black'
    #     elif ratioList[0] >=0.20:
    #         print("Non black")
    #         return 'non-black'

img = cv2.imread("test_inter4.png")

lower_red = np.array([0,160,120])
upper_red = np.array([200,255,255])

# gremio -> use a range that goes to light blue til very dark
lower_blue = np.array([110,50,50])
upper_blue = np.array([145,255,255])

rat, pre_img, out = DetectTeams(img, lower_red, upper_red)

out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
           
"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# plot multiple images
plt.subplots(1, 2, figsize=(20, 15))


plt.subplot(1, 2, 1), plt.imshow(pre_img, vmin = 0, vmax = 255)
plt.title('original')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(out, cmap = 'gray', vmin = 0, vmax = 255)
plt.title('out')
plt.xticks([]),plt.yticks([])

plt.show()
plt.close()



# gray, thresh = PreProcessing.GrayThresh(image)


#  # yellow range
# lower_yellow = np.array([21, 180, 64])
# upper_yellow = np.array([40, 200, 255])

# # red range
# lower_red = np.array([0,160,50])
# upper_red = np.array([10,255,255])

# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
