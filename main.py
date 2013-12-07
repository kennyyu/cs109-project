"""
Main driver
"""

import features
import learners
import reduction
import pandas as pd

topscores = {'liberal': 106, 'videos': 10341, 'gentlemanboners': 1619, 'books':
        4914, 'Music': 7286, 'politics': 15133, 'nba': 4108, 'pokemon': 3270,
        'funny': 9633, 'technology': 10848, 'conservative': 438, 'food': 3358,
        'WTF': 11107, 'worldnews': 10559, 'soccer': 2985, 'gaming': 16413,
        'aww': 7656, 'circlejerk': 3069, 'ladyboners': 1190, 'news': 10995,
        'television': 9274, 'science': 8965, 'nfl': 5416, 'pics': 19196,
        'movies': 93504}

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
