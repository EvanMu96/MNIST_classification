import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from utils import loadNPreprocess_data, load_sampled_subsets
from sklearn.externals import joblib

if sys.argv[1]=='all':
    train_image, train_label, test_image, test_label = loadNPreprocess_data()
elif sys.argv[1]=='sample':
    train_image, train_label, test_image, test_label = load_sampled_subsets(200)

if __name__ == "__main__":
    print("------------Start Training------------")
    # train neural network with warm start
    # original tutorial : https://scikit-learn.org/stable/modules/neural_networks_supervised.html#more-control-with-warm-start
    model = MLPClassifier(hidden_layer_sizes=(500,), solver='adam', random_state=1, max_iter=5, warm_start=True)
    prev_score = 0
    for i in range(3000):
        if i%10==0:
            print("[start training iteration {}]".format(i*5))
        model.fit(train_image, train_label)
        test_score = model.score(test_image, test_label)
        print("The score on test set is {0}".format(test_score))
        if (abs(test_score - prev_score)<0.001) and (test_score>0.98):
            joblib.dump(model, 'best_ANN_model_1h.pkl')
            print("The stop criterion is satisfied. Process ended and model saved")
            print("Testing Score is {0}".format(test_score))
            exit(0)
        prev_score = test_score