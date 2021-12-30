import os
import cv2
import time
import pickle
import argparse
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from svm import LinearSVM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img_path", required=False, default='data/temp/3_858.png', help="Path to test image")
    parser.add_argument("-m", "--model", choices=['sklearn', 'scratch'], required=True, help="Choose type of implementation")
    parser.add_argument("-p", "--model_path", required=False, default='model', help="Model save path")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("[INFO] extracting features...")
    image = cv2.imread(args.img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))

    hogFeature = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(4, 4), 
        transform_sqrt=True, 
        visualize=False,
        block_norm='L2')

    data = np.stack(hogFeature, axis=0)
    data = np.expand_dims(data, axis=0)
    data = np.hstack([data, np.ones((data.shape[0], 1))])
    print(data.shape)

    print("[INFO] loading classifier...")
    start = time.time()
    if args.model=='scratch':
        load_svm = LinearSVM()
        load_svm.load_weights(row = data.shape[1], col=8, path=args.model_path)
        label = load_svm.predict(data.reshape((1,-1)))
    else:
        load_svm = SGDClassifier()
        load_svm = pickle.load(open(os.path.join(args.save_path, 'svm_sklearn.sav'), 'rb'))
        clf = CalibratedClassifierCV(load_svm)
        label = clf.predict(data.reshape((1,-1)))
        score = clf.predict_proba(data.reshape((1,-1)))

    labels_list = ['cam nguoc chieu', 'cam dung va do', 'cam re', 'gioi han toc do', 'cam khac', 'nguy hiem', 'hieu lenh', 'negative']
    print("[INFO] predicting time: ", time.time()-start)
    print('[INFO] Predicting result: ', labels_list[int(label)])
    try:
        print('[INFO] Confident score: ', score)
    except:
        pass

if __name__ == '__main__':
    main()
    """
    mini zalo data
    0: cam nguoc chieu
    1: cam dung va do
    2: cam re
    3: gioi han toc do
    4: cam khac
    5: nguy hiem
    6: hieu lenh
    7: negative
    """