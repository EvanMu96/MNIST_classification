# evaluation program for helping TA checking homework
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from utils import loadNPreprocess_data, load_sampled_subsets, load_imbalanced_subsets
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.externals import joblib

# IMPORTANT PLEASE TURN OFF THE TRAIN FLAG
test_image, test_label = loadNPreprocess_data(train=False)

if __name__ == "__main__":
    print("----------Starting Evaluation----------")
    print("Here is the best perfomance Support Vector Machine, with auto gamma RBF kernel, C=100")
    svc = joblib.load('saved_best_rbf_SVM_model.pkl')
    # print the accuracy
    print("The accuarcy is {}".format(svc.score(test_image, test_label)))
    print("---------------------------------------")
    print("Here is the best Artificial Neural Network, with 3 hidden layer 1000, 500, 300 neurons respectively")
    ann = joblib.load('best_ANN_model.pkl')
    print("The accuarcy is {}".format(ann.score(test_image, test_label)))
    