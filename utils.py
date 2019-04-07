import numpy as np
import pandas as pd

# split training and test data into images and labels
def split_label_image(data):
    data_label = data[:, 0]
    # normalize pixel value into [0, 1], the reason: https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
    data_image = data[:, 1:]/255
    return data_label, data_image

# load entire dataset
def loadNPreprocess_data():
    train_data = pd.read_csv('data/train.csv').to_numpy()
    test_data = pd.read_csv('data/test.csv').to_numpy()
    train_label, train_image = split_label_image(train_data)
    test_label, test_image = split_label_image(test_data)
    return train_image, train_label, test_image, test_label

# random identically sample
def id_sample(data, sample_size):
    grouped_data = []
    # identically sample through entire dataset
    for i in range(10):
        grouped_data.append(data[data.label==i].sample(sample_size))
    return pd.concat(grouped_data) # return the balanced sampled set

# load balanced subsets
def load_sampled_subsets(sample_size):
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv').to_numpy()
    train_data = id_sample(train_data, sample_size).to_numpy()
    train_label, train_image = split_label_image(train_data)
    test_label, test_image = split_label_image(test_data)
    return train_image, train_label, test_image, test_label

# just for test
if __name__ == "__main__":
    print(load_sampled_subsets(1))