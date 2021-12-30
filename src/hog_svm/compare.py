import cv2
import numpy as np
from skimage.feature import hog
from matplotlib.pyplot import plot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
import argparse, logging, time
from imutils import paths
import pickle
from tqdm import tqdm
from hog import hog_descriptor_scratch
from utils import extract_feature


def train(descriptor=None):
    logging.info('Training SVM classifier with {} descriptor'.format(descriptor))
    
    model_path = 'models/hog_{}_classifier.sav'.format(descriptor)
        
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=False, default='data/svm/train', help="Path to the training dataset")
    ap.add_argument("-v", "--validation", required=False, default='data/svm/valid', help="Path to the validation dataset")
    args = vars(ap.parse_args())
    
    start = time.time()
    data_train, labels_train, _ = extract_feature(path=args["training"], method=descriptor)
    data_val, labels_val, _ = extract_feature(path = args["validation"], method=descriptor)
    
    logging.info("Finish extracting HoG features with {} descriptor. Total time: {}".format(descriptor, time.time()-start))
    start = time.time()
    logging.info("Training...")
    clf = SGDClassifier(learning_rate='optimal', loss='hinge', penalty='l2', alpha=0.001, max_iter=15000, verbose=False, n_jobs=-1, tol=1e-3, early_stopping=True)
    
    clf_with_prob = CalibratedClassifierCV(clf)
    clf_with_prob.fit(data_train, labels_train)
    
    logging.info("Training accuracy: {}".format( clf_with_prob.score(data_train, labels_train)))
    logging.info("Finish training SVM model. Total time: {}".format(time.time()-start))

    logging.info("Saving model...")
    pickle.dump(clf_with_prob, open(model_path, 'wb'))  
    logging.info('Validation accuracy: {}'.format(clf_with_prob.score(data_val, labels_val)))
    
    # print('Test accuracy on Scratch HoG extractor', model2.score(data2, labels))
    logging.info("Confusion matrix...")
    logging.info(confusion_matrix(clf_with_prob.predict(data_val), labels_val))

if __name__=='__main__':
    logging.basicConfig(filename='logs/hog_descriptor_compare.log', level=logging.INFO)
    logging.info('Comparing hog descriptor from scratch and scikit image library')
    
    train(descriptor='scratch')
    train(descriptor='skimage')