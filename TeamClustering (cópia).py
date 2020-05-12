import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from PreProcessing import PreProcessing
import seaborn as sns
sns.set_style('whitegrid')

class Colors:
    
    
    """"
    new approach:
        - identify the avg color value inside rectangle for each player
        - use a k means clustering to get which team does he belong to 
        - if we do not get good outputs, try to color quantization
        (https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html)
    """
    
    def GetAvg(image, k):
        # se o resultado for ruim, dá pra tentar tirar o fundo verde
        reshape = image.reshape((image.shape[0]*image.shape[1], 3))
        cluster = KMeans(n_clusters = k).fit(reshape)
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins = labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        rect = np.zeros((50, 300, 3), dtype=np.uint8)
        colors = sorted([(percent, color) for (percent, color) in zip(hist, cluster.cluster_centers_)])
        
        # por ora, fodase as porcentagens
        main_list = []
        for j in range(k):
            array = colors[j][1]
            main_list.append(array)
            
        main_array = np.vstack(main_list)
        lista = np.concatenate(main_array).ravel().tolist()
        
        # out: lista de k arrays, array de k arrays, lista com k*3 elementos
        return main_list, main_array, lista
        


    #input colors should be np.array
    def TeamsK(frame, k):
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gray, thresh = PreProcessing.GrayThresh(frame)
        
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
          
        kmeans = KMeans(n_clusters = 2) 
        ls = []
        ar = []
        value = []
        images = [] 
    
        for i in range(len(x_)):
            if(h_[i]>(1.2)*w_[i]):
                if(w_[i]>6 and h_[i]>6):
                    print('identified player {}'.format(i))
                    player_image = frame[y_[i]:y_[i]+h_[i], x_[i]:x_[i]+w_[i]]
                    images.append(player_image)
                    _, _, colors = Colors.GetAvg(player_image, k)
                    value.append(colors)
                    
                    
                    
                    
        
        return value, images
    



k = 3
img = cv2.imread("BraBel.png")
ls, imgs = Colors.TeamsK(img, k)

players = {}
for j in range(len(ls)):
    players['player' + str(j+1)] = ls[j]

df = pd.DataFrame(players, index = ['R1', 'G1', 'B1', 'R2', 'G2', 'B2', 'R3', 'G3', 'B3'])
df = df.T

# #gremio = 0, inter = 1
# df['Target'] = [0, 1, 1, 1, 0, 0, 0,1,0,0,0,0,1]

# bra = 0, bel = 1
df['Target'] = [1, 0, 0, 0, 1, 0, 1, 0, 0]


""" 
implementing K Means Clustering model

"""

kmeans = KMeans(n_clusters = 2)
kmeans.fit(df)
df['Labels'] = kmeans.labels_

sns.lmplot('R1', 'R2', hue = 'Labels', data = df, palette = 'coolwarm', fit_reg = False)








"""
//////////////////////////
/// Testing Function ////
/////////////////////////
 (\__/)  ||
 (•ㅅ•)  ||
 ( 　 づ || 
"""        

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
