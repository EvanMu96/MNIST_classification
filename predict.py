# evaluation program for helping TA checking homework
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from utils import loadNPreprocess_data, load_sampled_subsets, load_imbalanced_subsets
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.externals import joblib

if __name__ == "__main__":
    print("----------Starting Evaluation----------")