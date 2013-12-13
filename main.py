"""
Main driver
"""

import argparse
import features
import learners
import reduction
import pandas as pd
from sklearn import cross_validation
import json
import numpy as np
import unsupervised

"""
feature models names -> functions that return feature model instances
"""
FEATURES = {
    "1gram" : features.BagOfWordsModel,
    "2gram" : lambda: features.NGramModel(2),
    "3gram" : lambda: features.NGramModel(3),
    "4gram" : lambda: features.NGramModel(4),
    "5gram" : lambda: features.NGramModel(5),
    "6gram" : lambda: features.NGramModel(6),
    "7gram" : lambda: features.NGramModel(7),
    "8gram" : lambda: features.NGramModel(8),
    "9gram" : lambda: features.NGramModel(9),
    "10gram" : lambda: features.NGramModel(10),
    "100lda" : lambda: features.LdaFeatureModel(num_topics=100),
    "200lda" : lambda: features.LdaFeatureModel(num_topics=200),
    "500lda" : lambda: features.LdaFeatureModel(num_topics=500),
    "1000lda" : lambda: features.LdaFeatureModel(num_topics=1000),
    "1500lda" : lambda: features.LdaFeatureModel(num_topics=1500),
    "2000lda" : lambda: features.LdaFeatureModel(num_topics=2000),
}

"""
reducer names -> functions that take reduced dim as an arg
"""
REDUCERS = {
    "select" : reduction.SelectKBestReduction,
    "pca-linear" : lambda dim : reduction.KernelPCAReduction(dim, kernel='linear'),
    "pca-cosine" : lambda dim : reduction.KernelPCAReduction(dim, kernel='cosine'),
    "none" : lambda dim : reduction.NoopReduction(),
}

"""
learner names -> functions that return learner instances
"""
LEARNERS = {
    "nb" : learners.GaussianNBLearner,
    "svm-linear" : lambda: learners.SVMLearner(kernel='linear'),
    "svm-rbf" : lambda: learners.SVMLearner(kernel='rbf'),
    "svm-poly" : lambda: learners.SVMLearner(kernel='poly'),
    "knn" : lambda: learners.KNeighborsLearner(),
    "tree" : lambda: learners.DecisionTreeLearner(),
}

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
    arrays = dict((field, []) for field in fields)
    #arrays = {field:[] for field in fields}
    for line in file.readlines():
        data = json.loads(line)
        for field in fields:
            arrays[field].append(data[field])
    df = pd.DataFrame(arrays)
    file.close()
    return df

def compute_score(learner, X, Y):
    """
    get predictions for X, and compute the sum of the absolute
    differences between our predictions and true values
    """
    return np.mean(np.abs(learner.predict(X) - np.array(Y)))

def test_performance(df, model_name, learner_name, reducer_name, n_folds, dim):
    """
    Does cross validation on the dataframe, with n_folds.
    """
    model = FEATURES[model_name]()
    if model_name == "lda":
        reducer_name = "none"

    # transform the input data into feature vectors and labels
    X, y = model.make_training_xy(df)
    kfolds = cross_validation.KFold(len(df.index), n_folds)

    # for each of the folds, create a training and test set
    fold = 1
    test_errors = []
    train_errors = []
    for train_index, test_index in kfolds:
        reducer = REDUCERS[reducer_name](dim)
        learner = LEARNERS[learner_name]()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = list(np.asarray(y)[train_index]), list(np.asarray(y)[test_index])

        # Reduce the dimensionality of our training set
        reducer.fit(X_train, y_train)
        X_train_red = reducer.transform(X_train)

        # Train our learner on the reduced features
        learner.train(X_train_red, y_train)
        train_score = compute_score(learner, X_train_red, y_train)
        train_errors.append(train_score)

        # Apply the same dimensionality reduction to the test set's features
        # test the performance of the model on the test set
        X_test_red = reducer.transform(X_test)
        test_score = compute_score(learner, X_test_red, y_test)
        test_errors.append(test_score)

        print "--------------"
        print "TEST  ERROR " + str(fold) + ": " + str(model.y_to_label(df, [test_score]))
        print "TRAIN ERROR " + str(fold) + ": " + str(model.y_to_label(df, [train_score]))
        fold = fold + 1
    print "--------------"
    print "MEAN TEST  ERROR:", str(model.y_to_label(df, [np.mean(test_errors)]))
    print "MEAN TRAIN ERROR:", str(model.y_to_label(df, [np.mean(train_errors)]))

def clean_comment(s):
    s = s.lower()
    for c in ',./?;:\'\"[]{}`~!@#$%^&*()=+_\\|':
        s = s.replace(c, '')
    return s

parser = argparse.ArgumentParser("Run Upvote predictor. run with python -i")
parser.add_argument("subreddit", help="path to subreddit file", type=str)
parser.add_argument("model_name", help="feature model to use",
                    type=str, choices=FEATURES.keys())
parser.add_argument("reducer_name", help="reducer model to use",
                    type=str, choices=REDUCERS.keys())
parser.add_argument("learner_name", help="learner model to use",
                    type=str, choices=LEARNERS.keys())
parser.add_argument("--dim", help="reduced dimension size",
                    type=int, dest="dim", default=2000)
parser.add_argument("--folds", help="perform cross validation, num folds",
                    dest="folds", type=int, default=0)
parser.add_argument("--comments", help="path to comments file",
                    dest="comments", type=str, default="")
parser.add_argument("--clusters", help="cluster comments within subreddit",
                    dest="clusters", type=int, default=0)

def main(subreddit, comments, model_name, reducer_name, learner_name,
         dim, folds, clusters):
    model = FEATURES[model_name]()
    if model_name == "lda":
        reducer_name = "none"
    reducer = REDUCERS[reducer_name](dim)
    learner = LEARNERS[learner_name]()

    print "model: %s, reducer: %s, learner: %s, reduced dim: %d" \
        % (model_name, reducer_name, learner_name, dim)

    print "opening subreddit file:", subreddit
    df = load_subreddit(subreddit)
    subreddit_name = df["subreddit"][0]
    print "subreddit:", subreddit_name
    print "num rows:", len(df.index)
    print "max upvotes:", features.denormalize_scores([1.], subreddit_name)

    if folds > 0:
        print ">>>>> cross validating with %d folds" % folds
        test_performance(df, model_name, learner_name, reducer_name, folds, dim)
        print ">>>>>"

    # don't bother to produce the training set or reduce dimensionality
    # if we are not providede with a test file or cluster numbers
    if comments == "" and clusters == 0:
        return

    print ">>>>>>"
    print "making training data..."
    X_train, Y_train = model.make_training_xy(df)
    print "done"

    print "reducing dimensionality..."
    reducer.fit(X_train, Y_train)
    X_train_red = reducer.transform(X_train)
    print "done"

    if comments != "":
        print "training learner..."
        learner.train(X_train_red, Y_train)
        print "done"

        print "getting test data from %s ..." % comments
        testfile = open(comments, "rb")
        testdata = testfile.readlines()
        testdata = [line.strip() for line in testdata]
        testfile.close()
        new_df = pd.DataFrame({'body' : testdata,
                               'subreddit' : [subreddit_name] * len(testdata)})
        X_test = model.data_to_x(new_df)
        X_test_red = reducer.transform(X_test)
        print "done"

        print "predicting test labels..."
        Y_test = learner.predict(X_test_red)
        Y_upvotes = model.y_to_label(df, Y_test)
        print "done"

        print ""
        print ">>>>> RESULTS"
        for comment, upvote in zip(testdata, Y_upvotes):
            print upvote, comment
        print ">>>>>"
        print

    if clusters > 0:
        print ">>>>> CLUSTERING with %d clusters" % clusters
        unsupervised.cluster_within_subreddit(df, X_train_red, clusters)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
