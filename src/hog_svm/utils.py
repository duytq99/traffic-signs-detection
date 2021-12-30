import cv2
import numpy as np
from tqdm import tqdm
from imutils import paths
from skimage.feature import hog
from hog import hog_descriptor_scratch

def extract_feature(path, method = 'skimage'):
    print("[INFO] extracting training features from {}...".format(path))
    data = []
    labels = []
    filenames = []
    index = 0
    for imagePath in tqdm(paths.list_images(path)):
        index +=1
        make = imagePath.split("\\")[-2]
	
        # load the image, convert it to grayscale, and detect edges
        image = cv2.imread(imagePath)
        try:			
            gray = cv2.resize(image, (96, 96))
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            # extract Histogram of Oriented Gradients from the logo
            if method == 'skimage':
                hogFeature = hog(gray,orientations=9,pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True,visualize=False,block_norm='L2')
            else:
                try:
                    hogFeature = hog_descriptor_scratch(gray, cell_size=(8,8), orientations = 9, block_norm = 'L2', cells_per_block=(2,2), visualize = False, visualize_grad=False)
                except:
                    print("ERROR HERE")
            data.append(hogFeature)
            labels.append(make)
            filenames.append(imagePath)
        except:
            print(imagePath)

    data = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)
    print("[INFO] Feature shape: {}".format(data.shape))
    return data, labels, filenames


