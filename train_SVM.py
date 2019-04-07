import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from utils import loadNPreprocess_data

train_image, train_label, test_image, test_label = loadNPreprocess_data()

def experiment():
    # The neural network codes are not recommend to use because it has a better trainng stop citerion
    # the hidden layer neron numbers are 200, 300, 200 respectively
    # model = MLPClassifier(hidden_layer_sizes=(100, 200, 300), max_iter=4000, alpha=0.0001, solver='adam', random_state=1)
    # alternatively, here is the SVM model
    # l2 penalty, C = 100, linear kernel Support Vector Machine
    model = LinearSVC(C=100)
    model.fit(train_image, train_label)
    # scoring, evaluation with training and test data
    print("Training set score: {}".format(model.score(train_image, train_label)))
    test_score = model.score(test_image, test_label)
    print("Test set score: {}".format(test_score))
    return model, test_score

if __name__ == "__main__":
    # specify how many models should the program train
    print("-------------start training-------------")
    exp_number = 5
    test_scores = []
    best_model = None
    # do multiple training to select best model
    for i in range(exp_number):
        print("[ start iteration {} ]".format(i+1))
        model, score = experiment()
        if (i==0):
            best_model = model
            test_scores.append(score)
        if score > max(test_scores):
            best_model = model
            test_scores.append(score)
        else:
            test_scores.append(score)

    print("The mean acurracy among {0} models is {1}".format(exp_number, sum(test_scores)/len(test_scores)))
    print("The best model acurracy is {0}". format(max(test_scores)))
    # saving the best model into pickles file
    #joblib.dump(best_model, 'saved_best_ANN_model.pkl')
    # alternatively
    joblib.dump(best_model, 'saved_best_SVM_model.pkl')