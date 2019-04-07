import numpy as np
import pandas as pd

# load dataset
def loadNPreprocess_data():
    train_data = pd.read_csv('data/train.csv').to_numpy()
    test_data = pd.read_csv('data/test.csv').to_numpy()
    # split training and test data into images and labels
    train_label = train_data[:, 0]
    train_image = train_data[:, 1:]
    test_label = test_data[:, 0]
    test_image = test_data[:, 1:]
    # normalize pixel value into [0, 1], the reason: https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
    train_image = train_image/255
    test_image = test_image/255
    return train_image, train_label, test_image, test_label