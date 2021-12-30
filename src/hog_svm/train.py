import os
import time
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from skimage.feature import hog
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import SGDClassifier

from svm import LinearSVM
from utils import extract_feature


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plot
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=['sklearn', 'scratch'], required=True, help="Choose type of implementation")
    parser.add_argument("-t", "--training", required=False, default='data/svm/train', help="Path to the training dataset")
    parser.add_argument("-v", "--validation", required=False, default='data/svm/valid', help="Path to the validation dataset")
    parser.add_argument("-p", "--save_path", required=False, default='model', help="Model save path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    # initialize the data matrix and labels
    start = time.time()
    data_train, labels_train, _ = extract_feature(path=args.training)
    data_val, labels_val, _ = extract_feature(path = args.validation)
    print("[INFO] Finish extracting HoG features. Total time: {}".format(time.time()-start))

    print("[INFO] Training with %s svm model" % (args.model))
    tic = time.time()
    if args.model=='scratch':
        svm = LinearSVM()
        loss_hist = svm.train(
            data_train, 
            labels_train, 
            learning_rate=1e-2, 
            reg=0.001, 
            num_iters=15000, 
            verbose=False)
    else:
        svm = SGDClassifier(
            learning_rate='optimal', 
            loss='hinge', 
            penalty='l2', 
            alpha=0.001, 
            max_iter=15000, 
            verbose=False, 
            n_jobs=-1, 
            tol=1e-3, 
            early_stopping=True)
    toc = time.time()
    print ('[INFO] That took %fs' % (toc - tic))

    print("[INFO] Saving model...")
    if args.model=='scratch':
        svm.save_weights(path=os.path.join(args.save_path, 'svm_scratch.sav'))
        plt.plot(loss_hist)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.grid('on')
        plt.savefig("output/training_log_svm.png")
    else:
        pickle.dump(svm, open(os.path.join(args.save_path, 'svm_sklearn.sav'), 'wb')) 


    y_train_pred = svm.predict(data_train)
    print ('[INFO] Training accuracy: %f' % (np.mean(labels_train == y_train_pred), ))

    if args.model=='scratch':
        load_svm = LinearSVM()
        load_svm.load_weights(row = data_val.shape[1], col = len(os.listdir(args.validation)), path=os.path.join(args.save_path, 'svm_scratch.sav'))
        pred_label = load_svm.predict(data_val)
    else:
        load_svm = SGDClassifier()
        load_svm = pickle.load(open(os.path.join(args.save_path, 'svm_sklearn.sav'), 'rb'))
        pred_label = load_svm.predict(data_val)

    print('[INFO] Confusion matrix... \n', metrics.confusion_matrix(pred_label, labels_val))
    print('[INFO] Validation accuracy: {}'.format(metrics.accuracy_score(labels_val, pred_label)))

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    labels = ['cam nguoc chieu', 'cam dung va do', 'cam re', 'gioi han toc do', 'cam khac', 'nguy hiem', 'hieu lenh', 'negative']
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(load_svm, data_val, labels_val,
                                     display_labels=labels,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        plt.show()
        print(title)
        print(disp.confusion_matrix)

if __name__ == '__main__':
    main()
