ReadMe:

This script allows a user to cluster the various colors of the image to compress size. It is assumed that the user provides the following files:
        Image path: str
	absolute path to image files   
    
This script requires that `numpy`, `matplotlib`, `sklearn`, `scipy` and `sys be installed within the Python environment you are running this script in.

The script can be executed as follows
	python quantize_color.py image1_path image2_path …


For PCA, 
The script can be executed as follows
	python image_pca.py image1_path image2_path …