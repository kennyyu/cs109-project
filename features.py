from abc import ABCMeta, abstractmethod
import numpy as np
from gensim import corpora
from gensim.models import ldamodel
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer
from utils import *

topscores = {'Liberal': 106, 'videos': 10341, 'gentlemanboners': 1619, 'books':
        4914, 'Music': 7286, 'politics': 15133, 'nba': 4108, 'pokemon': 3270,
        'funny': 9633, 'technology': 10848, 'Conservative': 438, 'food': 3358,
        'WTF': 11107, 'worldnews': 10559, 'soccer': 2985, 'gaming': 16413,
        'aww': 7656, 'circlejerk': 3069, 'LadyBoners': 1190, 'news': 10995,
        'television': 9274, 'science': 8965, 'nfl': 5416, 'pics': 19196,
        'movies': 93504}

class AbstractFeatureModel(object):
    """
    Interface for all feature extractors. Extend this class to
    create new models for analyzing comment data.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def make_training_xy(self, data):
        """
        Extract a feature matrix X and value vector Y from the training data set.

        Args
        ----
        data : dataframe containing comments data and upvote data

        Returns
        -------
        X : numpy array (dims: ncomments x nfeatures)
            each row of X represents the features associated with each comment
        Y : numpy array (dims: ncomments)
            each entry corresponds to the value associated with each comment
            (e.g. normalized upvote score, subreddit id)

        Usage
        -----
        model = MyModel(param1, param2)
        X, Y = model.make_training(data)
        """
        pass

    @abstractmethod
    def data_to_x(self, new_data):
        """
        Extracts a feature matrix X from the new data set (for predictions), where
        the number of rows of X is the number of new data items in new_data

        Args
        ----
        new_data : dataframe containing comments data, but no label data (upvotes, subreddit)

        Returns
        -------
        X : feature matrix (num rows equal to number of entries in new_data)
        """
        pass

    @abstractmethod
    def y_to_label(self, data, Y):
        """
        Translates a y value back into its true representation (e.g. the
        denormalized upvote score, the subreddit name).

        Args
        ----
        data : dataframe containing comments data and upvote data

        Returns
        -------
        labels : The human readable label for the given Y values, len(labels) == len(Y)
        """
        pass

class BagOfWordsModel(AbstractFeatureModel):
    """
    Bag of words model. This is only an example. TODO
    """

    def __init__(self, min_df=0):
        self.vectorizer = CountVectorizer(min_df=min_df)

    def make_training_xy(self, data):
        X = self.vectorizer.fit_transform(data.body)
        X = X.tocsc()
        Y = normalize_scores(data.ups, data.subreddit[0])
        return X,Y

    def data_to_x(self, new_data):
        return self.vectorizer.transform(new_data.body)

    def y_to_label(self, data, Y):
        return denormalize_scores(Y, data.subreddit[0])

class NGramModel(AbstractFeatureModel):
    """
    n-gram model for analyzing text
    """

    def __init__(self, n, min_df=0):
        self.n = n
        self.vectorizer = CountVectorizer(ngram_range=(n,n), min_df=min_df)

    def make_training_xy(self, data):
        X = self.vectorizer.fit_transform(data.body)
        X = X.tocsc()
        Y = normalize_scores(data.ups, data.subreddit[0])
        return X, Y

    def data_to_x(self, new_data):
        return self.vectorizer.transform(new_data.body)

    def y_to_label(self, data, Y):
        return denormalize_scores(Y, data.subreddit[0])

class LdaFeatureModel(AbstractFeatureModel):
    """
    LDA Topic model
    """
    def __init__(self, num_topics=10, printing=True):
        self.lda = None
        self.num_topics = num_topics
        self.printing = printing


    def make_training_xy(self, data):
        # convert data to "docs" for lda
        docs = []
        for post in data.body:
            docs.append(post.split(" "))

        # make LDA model
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        self.lda = ldamodel.LdaModel(corpus, id2word=self.dictionary, num_topics=self.num_topics)

        if self.printing:
            print "LDA Topics:"
            for topic in xrange(self.num_topics):
                print "Topic #{}".format(topic),
                print self.lda.print_topic(topic)

        # make X and Y
        X = self.docs_to_lda_matrix(docs)
        Y = normalize_scores(data.ups, data.subreddit[0])

        return np.array(X), Y

    def data_to_x(self, new_data):
        # convert data to "docs" for lda
        docs = []
        for post in new_data.body:
            docs.append(post.split(" "))

        # make X
        X = self.docs_to_lda_matrix(docs)

        return np.array(X)

    def y_to_label(self, data, Y):
        return denormalize_scores(Y, data.subreddit[0])

    def docs_to_lda_matrix(self, docs):
        res = []
        for doc in docs:
            row = [0.0] * self.num_topics
            pred = self.lda[self.dictionary.doc2bow(doc)]

            # iterate over topic prediction tuples
            for t in pred:
                row[t[0]] = t[1]

            # add row
            res.append(row)
        return res



class CooccurenceModel(AbstractFeatureModel):
    """
    Cooccurence model for analyzing text
    """

    def __init__(self, min_df=0):
        # build bag of words model
        self.bow_model = BagOfWordsModel(min_df)
        pass

    def make_training_xy(self, data):
        # get X/Y from bow_model
        bow_X, bow_Y = self.bow_model.make_training_xy(data)

        num_rows = bow_X.shape[0]
        num_features = bow_X.shape[1]

        # store sparse rows in a list
        rows = []

        # iterate over sparse rows
        for i in xrange(num_rows):
            # multiply X with itself to get coccurrence matrix
            bow_X_col = bow_X.transpose(copy=True)

            # reshape into a single row, and add to rows array
            cooc_matrix_row = coo_reshape(bow_X_col.getcol(0) * bow_X.getrow(0), (1, num_features * num_features)).tocsc()

            rows.append(cooc_matrix_row)

        # stack rows
        cooc_matrix = vstack(rows)

        # return TODO: should we remove duplicates? e.g. A/B and B/A?
        return cooc_matrix, bow_Y

    def data_to_x(self, new_data):
        # get counts from bow model
        box_X = self.bow_model.data_to_x(new_data)
        # multiply X with itself to get coccurrence matrix
        bow_X_col = bow_X.transpose(copy=True)

        # reshape into a single row, and add to rows array
        cooc_matrix_row = coo_reshape(bow_X_col.getcol(0) * bow_X.getrow(0), (1, num_features * num_features)).tocsc()

        # return
        return cooc_matrix_row

    def y_to_label(self, data, Y):
        return denormalize_scores(Y, data.subreddit[0])

"""
normalize_scores
    Normalizes the score based on the max upvotes in the given subreddit.

    @param: ups (array of upvote scores), subreddit (name of subreddit)
    @ret: array of normalized scores
"""
def normalize_scores(ups, subreddit):
    return [float(x)/topscores[subreddit] for x in ups]

def denormalize_scores(norms, subreddit):
    return [x * topscores[subreddit] for x in norms]

