from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.misc import imread
import numpy as np
import sys


class compressImage:
    def __init__(self, image, label):
        self.image = image
        self.w, self.h, self.d = self.image.shape
        self.components = [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.99
        ]
    
    def pca_image_green(self):
        image_array = np.array(self.image)
        green_component = image_array[:,:,1]
        for index, components in enumerate(self.components):
            # Green
            green_image_reshape = np.reshape(green_component, (self.w, self.h))
            green_image_pca = PCA(components).fit(green_image_reshape)
            green_image_compressed = green_image_pca.transform(green_image_reshape)
            green_temp = green_image_pca.inverse_transform(green_image_compressed)
            green_image_compressed_reshape = np.reshape(green_temp, (self.w, self.h))
            
            plt.figure(index + 1)
            plt.clf()
            label = "# of components: " + str(components)
            plt.title(label)
            plt.imshow(green_image_compressed_reshape)
            
            
            label = "green_image_" + str(index)
            
            plt.savefig(label)
    
    def pca_image(self):
        image_array = np.array(self.image)
        red_component = image_array[:,:,0]
        green_component = image_array[:,:,1]
        blue_component = image_array[:,:,2]
        
        
        for index, components in enumerate(self.components):
            # Red
            red_image_reshape = np.reshape(red_component, (self.w, self.h))
            red_image_pca = PCA(components).fit(red_image_reshape)
            red_image_compressed = red_image_pca.transform(red_image_reshape)
            red_temp = red_image_pca.inverse_transform(red_image_compressed)
            red_image_compressed_reshape = np.reshape(red_temp, (self.w, self.h))
            
            # Green
            green_image_reshape = np.reshape(green_component, (self.w, self.h))
            green_image_pca = PCA(components).fit(green_image_reshape)
            green_image_compressed = green_image_pca.transform(green_image_reshape)
            green_temp = green_image_pca.inverse_transform(green_image_compressed)
            green_image_compressed_reshape = np.reshape(green_temp, (self.w, self.h))
            
            # Blue
            blue_image_reshape = np.reshape(blue_component, (self.w, self.h))
            blue_image_pca = PCA(components).fit(blue_image_reshape)
            blue_image_compressed = blue_image_pca.transform(blue_image_reshape)
            blue_temp = blue_image_pca.inverse_transform(blue_image_compressed)
            blue_image_compressed_reshape = np.reshape(blue_temp, (self.w, self.h))
            
            
            image_compressed = np.dstack((
                red_image_compressed_reshape,
                green_image_compressed_reshape,
                blue_image_compressed_reshape
            ))
            plt.figure(index + 1)
            plt.clf()
            label = "# of components: " + str(components)
            plt.title(label)
            plt.imshow(image_compressed)
            plt.show()
            
    
def main():
    image_path = []
    for index in range(1, len(sys.argv)):
        image_path.append(sys.argv[index])
        
    for item in image_path:
        image = imread(item)
        label = item
        obj = compressImage(image, label)
        obj.pca_image_green()

if __name__ == '__main__':
    main()