"""
Main driver
"""


import features
import learners
import reduction
import pandas as pd
from sklearn import cross_validation
import json
import numpy as np

"""
Data format:

{
  "body": string
  "post_ups": int
  "subreddit_id": string
  "created": float (timestamp)
  "downs": int
  "author": string
  "post_net": int
  "subreddit": string
  "post_id": string
  "post_downs": int
  "net": int
  "ups": int
  "id": string
  "post_created": float
}
"""
FIELDS = ["body", "post_ups", "subreddit_id", "created", "downs",
          "author", "post_net", "subreddit", "post_id", "post_downs",
          "net", "ups", "id", "post_created"]

def load_subreddit(filename, fields=FIELDS):
    """
    Loads the subreddit with the filename and returns
    a dataframe where the column names are the fields
    in the json object.
    """
    file = open(filename, "rb")
    arrays = {field:[] for field in fields}
    for line in file.readlines():
        data = json.loads(line)
        for field in fields:
            arrays[field].append(data[field])
    df = pd.DataFrame(arrays)
    file.close()
    return df

if __name__ == "__main__":
    #model = features.BagOfWordsModel(tfidf)
    model = features.NGramModel(2)
    #model = features.CooccurenceModel()
    #reducer = reduction.KernelPCAReduction(2)
    reducer = reduction.SelectKBestReduction(10000)
    learner = learners.GaussianNBLearner()
#    learner = learners.MultiNBLearner(nbuckets=int(features.denormalize_scores([1.], 'Liberal')[0]))
#    learner = learners.SVMLearner(kernel='linear')

    data_file = "data/Liberal.txt"
    df = load_subreddit(data_file)
    df = df.head(10)
    print df.head(5)
    print "num rows:", len(df.index)
    print "max up:", features.denormalize_scores([1.], 'Liberal')

    # Make the training set
    print "making training data..."
    X_train, Y_train = model.make_training_xy(df)

    # Reduce the dimensionality of our training set
    print "reducing dimensionality..."
    reducer.fit(X_train, Y_train)
    X_train_red = reducer.transform(X_train)

    # Train our learner
    print "training our learner..."
    learner.train(X_train_red, Y_train)

    # Get test data/data from user
    words = ['pop off', 'hop hop pop', 'republican good', 'pro life',
             'Mitt Romney', 'stupid Republican',
             'I hate same sex marriage', 'I hate guns',
             'Barack Obama', 'the Senate', 'bipartisanship',
             'poor people', 'minimum wage', 'healthcare', 'obamacare',
             'obamacare sucks', 'obamacare is great',
             'need obamacare']
    words = [s.lower() for s in words]
    new_df = pd.DataFrame({'body' : words,
                           'subreddit' : ['Liberal'] * len(words)})
    print "getting some test data..."
    X_test = model.data_to_x(new_df)
    X_test_red = reducer.transform(X_test)

    # Use our learner to predict the new data's label
    print "predicting test label..."
    Y_test = learner.predict(X_test_red)
    new_label = model.y_to_label(df, Y_test)

    for word, label in zip(words, new_label):
        print label, word

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
