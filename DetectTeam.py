import cv2
import numpy as np
from matplotlib import pyplot as plt

class Colors:
    
    
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
        
        """"output:
            image_nofield -> preprocessed image
            out -> processed image
            ratio -> out color range processed image / out color range preprocessed image
                the greater the ratio, greater the prob that the player belongs to team A
            main out var -> team
            team = 1 -> belongs to team A
            team = 0 -> does not belong 
            
        """
    
        
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
        tot_pix = Colors.count_noncolor_np(image_nofield)
        color_pix = Colors.count_noncolor_np(output)
        ratio = color_pix/tot_pix
        
        team = 0
        
        if ratio > 0.1:
            team = 1
        else:
            team = 0
    
        return image_nofield, output, ratio, team


           
"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        
# img = cv2.imread("BraBel.png")

# lower_red = np.array([140,100,160])
# upper_red = np.array([190,255,255])

# # gremio -> use a range that goes to light blue til very dark
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([145,255,255])

# img_in, img_out, out, team = Colors.DetectTeams(img, lower_red, upper_red)


# # plot multiple images
# plt.subplots(1, 2, figsize=(20, 15))


# plt.subplot(1, 2, 1), plt.imshow(img_in, vmin = 0, vmax = 255)
# plt.title('original')
# plt.xticks([]),plt.yticks([])

# plt.subplot(1, 2, 2), plt.imshow(img_out, vmin = 0, vmax = 255)
# plt.title('out')
# plt.xticks([]),plt.yticks([])

# plt.show()
# plt.close()



# gray, thresh = PreProcessing.GrayThresh(image)


#  # yellow range
# lower_yellow = np.array([21, 180, 64])
# upper_yellow = np.array([40, 200, 255])

# # red range
# lower_red = np.array([0,160,50])
# upper_red = np.array([10,255,255])

# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
