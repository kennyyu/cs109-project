import numpy as np
import features
from main import load_subreddit

def make_ngram_freq(X_train, model):
    """
    Returns a list of (ngram, score) sorted by score, where
    - ngram is a string representing the ngram
    - score is the sum of all rows of X_train for that ngram
    """
    # sum column-wise of X_train to get features -> count for that feature
    X_freq = X_train.sum(0)
    X_freq = np.array(X_freq)[0].tolist()

    # get original n_grams
    ngrams = model.vectorizer.get_feature_names()

    # sort by highest frequency
    ngram_freq = zip(ngrams, X_freq)
    ngram_freq = sorted(ngram_freq, key=lambda (_,v) : v, reverse=True)
    return ngram_freq

def find_next_word(first, ngram_freq, random=False):
    """
    assumes ngram_freq is already reverse sorted by score

    given the first word, finds the highest scored ngram whose first
    word is `first`, and returns the next word of that ngram.
    """
    # get only ngrams whose first word matches first
    filtered_ngrams = filter(lambda (ngram, v): ngram.split()[0] == first,
                             ngram_freq)
    if len(filtered_ngrams) == 0:
        return ""

    # if we use randomness, we use the ngram's score
    # as part of a multinomial distribution so that we
    # are more likely to sample from ngrams with higher score
    if random:
        probs = [score for (word,score) in filtered_ngrams]
        sum_score = float(sum(probs))
        probs = [score / sum_score for score in probs]
        result = np.random.multinomial(n=1, pvals=probs, size=1)
        index = result[0].tolist().index(1)
        best_ngram = filtered_ngrams[index][0]
        return best_ngram.split()[1]

    # return second word of ngram
    best_ngram = filtered_ngrams[0][0]
    next = best_ngram.split()[1]

    # to avoid infinite recursion, make sure the next word
    # is different from the current word
    if next != first or len(filtered_ngrams) == 1:
        return next
    best_ngram = filtered_ngrams[1][0]
    return best_ngram.split()[1]

def build_sentence(start, nwords, ngram_freq, random=False):
    """
    Builds a sentence of length `nwords` with a starting word `start`
    using the most likely next word based on ngram scores in ngram_freq.

    Returns the sentence as a list of words
    """
    sentence = [start]
    for _ in range(nwords):
        current = sentence[-1]
        next = find_next_word(current, ngram_freq, random=random)
        if len(next) == 0:
            return sentence
        sentence.append(next)
    return sentence

if __name__ == '__main__':
    model = features.NGramModel(2)

    data_file = "data/Liberal"
    df = load_subreddit(data_file)

    # Make the training set
    print "making training data..."
    X_train, Y_train = model.make_training_xy(df)

    # Build most likely sentence with a root word
    ngram_freq = make_ngram_freq(X_train, model)
    nwords = 20

    def print_trial(start, nwords, random):
        print ">>> start: %s, length: %d, random: %s" % (start, nwords, str(random))
        sentence = build_sentence(start, nwords, ngram_freq, random=random)
        print " ".join(sentence)

    print_trial("obama", nwords, False)
    print_trial("obama", nwords, True)
    print_trial("liberal", nwords, False)
    print_trial("liberal", nwords, True)
    print_trial("liberals", nwords, False)
    print_trial("liberals", nwords, True)
    print_trial("republican", nwords, False)
    print_trial("republican", nwords, True)
    print_trial("republicans", nwords, False)
    print_trial("republicans", nwords, True)

