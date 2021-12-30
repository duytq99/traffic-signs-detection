import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from imutils import paths
from skimage.feature import hog
from sklearn import metrics
from svm import LinearSVM
from utils import extract_feature


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plot
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

trainPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage2_classifier\train'
valPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\stage2_classifier\valid'
modelPath = 'models/compare_scratch_model.sav'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=False, default=trainPath, help="Path to the training dataset")
    parser.add_argument("-v", "--validation", required=False, default=valPath, help="Path to the validation dataset")
    return parser.parse_args()


def main():
    args = parse_arguments()
    # initialize the data matrix and labels
    start = time.time()
    data_train, labels_train, _ = extract_feature(path=args.training)
    data_val, labels_val, filenames_val = extract_feature(path = args.validation)
    print("[INFO] Finish extracting HoG features. Total time: {}".format(time.time()-start))

    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(data_train, labels_train, learning_rate=1e-2, reg=0.001, num_iters=15000, verbose=False)
    toc = time.time()
    print ('[INFO] That took %fs' % (toc - tic))

    svm.save_weights(path = modelPath)

    # A useful debugging strategy is to plot the loss as a function of
    # iteration number:
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.grid('on')
    plt.savefig("figures/training_log.png")
    # plt.show()

    # Write the LinearSVM.predict function and evaluate the performance on both the
    # training and validation set
    y_train_pred = svm.predict(data_train)
    print ('[INFO] Training accuracy: %f' % (np.mean(labels_train == y_train_pred), ))

    load_svm = LinearSVM()
    load_svm.load_weights(row = data_val.shape[1], col = len(os.listdir(valPath)), path = r'models/compare_scratch_model.sav')
    predLabel = load_svm.predict(data_val)

    print('[INFO] Confusion matrix... \n', metrics.confusion_matrix(predLabel, labels_val))
    print('[INFO] Validation accuracy: {}'.format(metrics.accuracy_score(labels_val,predLabel)))

    # Plot non-normalized confusion matrix
    # from sklearn.metrics import plot_confusion_matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #                   ("Normalized confusion matrix", 'true')]
    # labels = ['cam nguoc chieu', 'cam dung va do', 'cam re', 'gioi han toc do', 'cam khac', 'nguy hiem', 'hieu lenh', 'negative']
    # for title, normalize in titles_options:
    #     disp = plot_confusion_matrix(load_svm, data_val, labels_val,
    #                                  display_labels=labels,
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=normalize)
    #     disp.ax_.set_title(title)

    #     print(title)
    #     print(disp.confusion_matrix)

    # from sklearn.metrics import plot_confusion_matrix

if __name__ == '__main__':
    main()
