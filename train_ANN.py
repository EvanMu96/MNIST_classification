import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from utils import loadNPreprocess_data, load_sampled_subsets, load_imbalanced_subsets
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.externals import joblib

if sys.argv[1]=='all':
    train_image, train_label, test_image, test_label = loadNPreprocess_data()
elif sys.argv[1]=='sample':
    train_image, train_label, test_image, test_label = load_sampled_subsets(200)
elif sys.argv[1]=='biased':
    biased_class = int(sys.argv[2])
    train_image, train_label, test_image, test_label = load_imbalanced_subsets(biased_class, 200)

if __name__ == "__main__":
    print("------------Start Training------------")
    # train neural network with warm start
    # original tutorial : https://scikit-learn.org/stable/modules/neural_networks_supervised.html#more-control-with-warm-start
    model = MLPClassifier(hidden_layer_sizes=(500,), solver='adam', random_state=1, max_iter=50, warm_start=True)
    prev_score = 0
    for i in range(100):
        if i%10==0:
            print("[start training iteration {}]".format(i*5))
        model.fit(train_image, train_label)
        test_score = model.score(test_image, test_label)
        print("The score on test set is {0}".format(test_score))
        if (abs(test_score - prev_score)<0.0005) and (test_score>0.90):
            #joblib.dump(model, 'best_ANN_model_imba.pkl')
            print("The stop criterion is satisfied. Process ended and model saved")
            print("Testing Score is {0}".format(test_score))
            y_pred = model.predict(test_image)
            pre = precision_score(test_label, y_pred, average=None)
            rec = recall_score(test_label, y_pred, average=None)
            acr = accuracy_score(test_label, y_pred)
            print("The accuracy for each class is {}".format(acr))
            print("The precision for each class is {}".format(pre))
            print("The recall for each class is {}".format(rec))
            exit(0)
        prev_score = test_score