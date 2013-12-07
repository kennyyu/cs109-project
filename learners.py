from abc import ABCMeta, abstractmethod
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

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
        self.nb.fit(X.toarray(), Y)

    def predict(self, X):
        return self.nb.predict(X.toarray())

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

