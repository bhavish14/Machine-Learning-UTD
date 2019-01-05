import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from scipy.misc import imread
import sys


class ImageQuantization:
    
    def __init__(self, image, colors, image_name):
        self.image = image
        self.image_name = image_name
        self.colors = colors        
        self.compressed = []
        

    def rebuild_image(self, codebook, labels, width, height, index):
        """
            Recreate the (compressed) image from the code book & labels
        """
        depth = codebook.shape[1]
        image = np.zeros((width, height, depth))
        label_idx = 0
        for i in range(width):
            for j in range(height):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        print (image.shape)
        return image
    
    def cluster_image(self):
        # Converting into range [0,1] from [0, 255]
        image = np.array(self.image, dtype = np.float64) / 255
        w, h, d = tuple(image.shape)

        image_arr = np.reshape(image, (w * h, d))

        image_subset = shuffle(image_arr, random_state = 0)[:1000]
    
        for index, num_clusters in enumerate(self.colors):
            clf = KMeans(
                n_clusters = num_clusters,
                random_state = 0
            )
            clf.fit(image_subset)
        
            predictions = clf.predict(image_arr)
            codebook_random = shuffle(image_arr, random_state=0)[:num_clusters]

            labels_random = pairwise_distances_argmin(codebook_random,
                                                      image_arr,
                                                      axis=0
            )
            label = "Compressed image (" + str(num_clusters) + " colors)"
            self.show_image(index, self.rebuild_image(codebook_random, labels_random, w, h, index), label, self.image_name)


            
    
    def show_image(self, id, image, label, image_name):
        plt.figure(id)
        plt.clf()
        plt.axis('off')
        plt.title(label)
        plt.imshow(image)
        save_name = "image_" + str(id) + "_" + str(image_name)
        plt.savefig(save_name)
        
def main():
    image_path = []
    for index in range(1, len(sys.argv)):
        image_path.append(sys.argv[index])

    num_clusters = [8, 16, 32, 64, 128, 256]

    for path in image_path:
        image = imread(path)  
        image_name = path  
        obj = ImageQuantization(image, num_clusters, image_name)
        obj.cluster_image()

if __name__ == '__main__':
    main()
