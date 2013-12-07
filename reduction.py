from abc import ABCMeta, abstractmethod
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest, chi2, f_regression

class AbstractReduction(object):
    """
    Interface for dimensionality reduction. Extend this for new models for
    reducing dimensionality of a dataset.

    Usage
    -----
    reducer = MyReduction(...)
    reducer.fit(X) # call this only once!
    X_reduced = reducer.transform(X)
    other_x_reduced = reducer.transform(other_x)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def n_components(self):
        """
        Return the target reduced dimension size.
        """
        pass

    @abstractmethod
    def fit(self, X, Y=None):
        """
        Fit the estimator to this data set. This must be called exactly ONCE
        to have consistent results when we call transform().

        Args
        ----
        X : feature matrix (n_comments x n_features)
        Y : values to fit to

        Returns
        -------
        Nothing
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Transforms the feature matrix X into a feature matrix of smaller size

        Args
        ----
        X : feature matrix (n_comments x n_features)

        Returns
        -------
        X_reduced : reduced feature matrix (n_comments x n_components)
        """
        pass

class KernelPCAReduction(AbstractReduction):
    """
    Use kernel PCA to reduce dimensionality

    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
    """

    def __init__(self, n_components, **kwargs):
        self.pca = KernelPCA(n_components=n_components, **kwargs)
        self.n_components = n_components

    def n_components(self):
        return self.n_components

    def fit(self, X, Y=None):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

class SelectKBestReduction(AbstractReduction):
    """
    Select K Best features using chi metric

    http://stackoverflow.com/questions/10098533/implementing-bag-of-words-naive-bayes-classifier-in-nltk
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
    """

    def __init__(self, n_components, score_func=lambda X, y: f_regression(X, y, center=False)):
        self.select = SelectKBest(score_func=score_func, k=n_components)
        self.n_components = n_components

    def n_components(self):
        return self.n_components

    def fit(self, X, Y=None):
        self.select.fit(X, Y)

    def transform(self, X):
        return self.select.transform(X)
