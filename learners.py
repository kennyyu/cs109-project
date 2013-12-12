from abc import ABCMeta, abstractmethod
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class AbstractLearner(object):
    """
    Interface for all learning models. Extend this class to create new learners.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, X, Y):
        """
        Trains our model.

        Args
        ----
        X : feature matrix. Each row is a feature vector of a datum
            (e.g. feature vector for a comment)
        Y : value vector. Each entry corresponds to the value associated
            with the corresponding row in X.

        num rows of X == length of Y

        Returns
        -------
        Does NOT return anything. train() will have side effects on the object
        and maintain state so that predict() can be called.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts the values for the feature matrix X. This MUST be called
        after train().

        Args
        ----
        X : feature matrix. Each row is a feature vector of a datum

        Returns
        -------
        Y : predicted values for each row of X
        """
        pass

class GaussianNBLearner(AbstractLearner):
    """
    Gaussian Naive Bayes Learner

    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

    We need to use X.toarray() because those functions expect dense arrays.
    """

    def __init__(self):
        self.nb = GaussianNB()

    def train(self, X, Y):
        if hasattr(X, 'toarray'):
            self.nb.fit(X.toarray(), Y)
        else:
            self.nb.fit(X, Y)

    def predict(self, X):
        if (hasattr(X, "toarray")):
            return self.nb.predict(X.toarray())
        else:
            return self.nb.predict(X)

    def score(self, X, Y):
        return np.mean(np.abs(self.nb.predict(X) - np.array(Y)))

class SVMLearner(AbstractLearner):
    """
    Support Vector Machine Learner for regression (continuous
    labels as opposed to discrete labels)

    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    def __init__(self, **kwargs):
        self.svr = SVR(**kwargs)

    def train(self, X, Y):
        self.svr.fit(X, Y)

    def predict(self, X):
        return self.svr.predict(X)

class KNeighborsLearner(AbstractLearner):
    """
    Learner using k-nearest neighbors
    """

    def __init__(self, **kwargs):
        self.knn = KNeighborsRegressor(**kwargs)

    def train(self, X, Y):
        self.knn.fit(X, Y)

    def predict(self, X):
        return self.knn.predict(X)

class MultiNBLearner(AbstractLearner):
    """
    Multinomial Naive Bayes Learner

    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

    """

    def __init__(self, nbuckets, **kwargs):
        self.nb = MultinomialNB(**kwargs)
        self.nbuckets = nbuckets

    def train(self, X, Y):
        newY = [0 for _ in range(len(Y))]
        for i, y in enumerate(Y):
            bucket = np.floor(y * self.nbuckets)
            newY[i] = bucket
        self.nb.fit(X, np.array(newY))

    def predict(self, X):
        return self.nb.predict(X)

class DecisionTreeLearner(AbstractLearner):
    """
    Decision Tree Regressor

    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    def __init__(self, **kwargs):
        self.tree = DecisionTreeRegressor(**kwargs)

    def train(self, X, Y):
        if hasattr(X, 'toarray'):
            self.tree.fit(X.toarray(), Y)
        else:
            self.tree.fit(X, Y)

    def predict(self, X):
        if (hasattr(X, "toarray")):
            return self.tree.predict(X.toarray())
        else:
            return self.tree.predict(X)

