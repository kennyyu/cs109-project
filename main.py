"""
Main driver
"""

import features
import learners
import reduction
import pandas as pd

if __name__ == "__main__":
    model = features.BagOfWordsModel(1)
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
