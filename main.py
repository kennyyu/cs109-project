"""
Main driver
"""

import features
import learners
import reduction
import pandas as pd
from sklearn import cross_validation

if __name__ == "__main__":
    model = features.BagOfWordsModel()
    #model = features.CooccurenceModel()
    reducer = reduction.KernelPCAReduction(2)
    learner = learners.GaussianNBLearner()

    # TODO: use real data
    df = pd.DataFrame({'body' : ['Hop on pop', 'Hop off pop', 'Hop Hop hop'],
                       'ups' : [0, 1, 0],
                       'subreddit' : ['liberal', 'liberal', 'liberal']})
    

    # Make the training set
    X_train, Y_train = model.make_training_xy(df)
    
    # Reduce the dimensionality of our training set
    reducer.fit(X_train)
    X_train_red = reducer.transform(X_train)

    # Train our learner
    learner.train(X_train_red, Y_train)

    # Get test data/data from user
    new_df = pd.DataFrame({'body' : ['pop off', 'hop hop pop'],
                           'liberal' : ['liberal', 'liberal']})
    X_test = model.data_to_x(new_df)
    X_test_red = reducer.transform(X_test)

    # Use our learner to predict the new data's label
    Y_test = learner.predict(X_test_red)
    new_label = model.y_to_label(df, Y_test)
    print new_label 

    # see how well this model generalizes
    # test_performance(df)
   
def test_performance(df):
    # transform the input data into feature vectors and labels
    X, Y = model.make_training_xy(df)
    kfolds = cross_validation.KFold(len(df.body), n_folds)
    
    # for each of the folds, create a training and test set
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], y[test_index]

        # Reduce the dimensionality of our training set
        reducer.fit(X_train)
        X_train_red = reducer.transform(X_train)

        # Train our learner on the reduced features
        learner.train(X_train_red, Y_train)

        # Apply the same dimensionality reduction to the test set's features
        # test the performance of the model on the test set
        X_test_red = reducer.transform(X_test)
        score = learner.score(X_test_red, y_test)

        print
        print "--------------"
        print "TRAINING INDEX:" + str(train_index)
        print "TEST INDEX:    " + str(test_index)
        print "SCORE:         " + str(score)
        print "--------------"
        print
