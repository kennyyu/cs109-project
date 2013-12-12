"""
CONFIG
======
Starts the server for upvote and sentence prediction. To run:

    python server.py config.json

Keys in config.json should exactly match the args to the

    init_* functions defined below.

SENTENCE PREDICTION
===================
To get a sentence prediction, send a GET request to /sentence?word=WORD.
That will return the next word (NO NEW LINE).
e.g.
   COMMAND:  curl --request GET "localhost:18888/sentence?word=office"
   RESPONSE: defeating

UPVOTE PREDICTION
=================
To get an upvote prediction, send a GET request to /up?comment=COMMENT
That will return the upvotes as a float. COMMENT must be query-formatted:
e.g.
   COMMAND:  curl --request GET "localhost:18888/up?comment=obama%02is%02awesome"
   RESPONSE: 31.0
"""

import argparse
import tornado.ioloop
import tornado.web
import json

import features
import reduction
import learners
from sentence import *
from main import *

SENT_NGRAM = 6
SENT_DATAFILE = "data/Liberal"
SENT_RANDOM = False
SENT_NGRAM_FREQ = None

UP_DF = None
UP_FEATURE = None
UP_REDUCER = None
UP_LEARNER = None
UP_DIM = 1000

def init_sentence(subreddit, ngram, random):
    """
    Initializes the ngram model for sentence prediction
    """
    global SENT_DATAFILE, SENT_NGRAM, SENT_RANDOM, SENT_NGRAM_FREQ

    SENT_DATAFILE = subreddit
    SENT_NGRAM = ngram
    SENT_RANDOM = random

    model = features.NGramModel(SENT_NGRAM)
    print "N used for ngram: %d" % SENT_NGRAM

    df = load_subreddit(SENT_DATAFILE)
    print "loaded: %s" % SENT_DATAFILE

    # Make the training set
    print "making training data..."
    X_train, Y_train = model.make_training_xy(df)

    # Build most likely sentence with a root word
    SENT_NGRAM_FREQ = make_ngram_freq(X_train, model)
    print "sentence init done"

def train_upvotes(df, model, reducer, learner):
    """
    Generates a training set, reduces dimensionality, and train on reduced
    dimensionality training set.
    """
    X_train, Y_train = model.make_training_xy(df)
    reducer.fit(X_train, Y_train)
    X_train_red = reducer.transform(X_train)
    learner.train(X_train_red, Y_train)

def predict_upvotes(comment, df, model, reducer, learner):
    """
    Return the predicted number of upvotes
    """
    comment = [clean_comment(comment)]
    new_df = pd.DataFrame({'body' : comment, 'subreddit' : [df['subreddit'][0]]})
    X_test = model.data_to_x(new_df)
    X_test_red = reducer.transform(X_test)
    Y_test = learner.predict(X_test_red)
    upvotes = model.y_to_label(df, Y_test)
    return upvotes[0]

def init_upvote(subreddit, feature, ngram, lda, reducer, learner, dim):
    """
    initializes the models for predicting upvotes
    """
    global UP_DF, UP_FEATURE, UP_REDUCER, UP_LEARNER, UP_DIM

    print "subreddit file:", subreddit
    UP_DF = load_subreddit(subreddit)
    name = UP_DF["subreddit"][0]
    print "subreddit:", name
    print "num rows:", len(UP_DF.index)
    print "max up:", features.denormalize_scores([1.], name)

    if feature == "ngram":
        UP_FEATURE = features.NGramModel(ngram)
    elif feature == "lda":
        UP_FEATURE = features.LdaFeatureModel(lda)
    UP_REDUCER = REDUCERS[reducer](dim)
    UP_LEARNER = LEARNERS[learner]()
    UP_DIM = dim

    print "training models..."
    train_upvotes(UP_DF, UP_FEATURE, UP_REDUCER, UP_LEARNER)
    print "upvote init done"

class SentenceHandler(tornado.web.RequestHandler):
    """
    Handler that takes a word and outputs the next word
    """
    def set_default_headers(self):
        #TODO change this to your own domain
        self.set_header("Access-Control-Allow-Origin", "http://localhost:8000")

    def get(self):
        word = self.get_argument("word", "")
        word = clean_comment(word)
        if word == "":
            next = ""
        else:
            next = find_next_word(word, SENT_NGRAM_FREQ, random=SENT_RANDOM)
        print "[X] SENTENCE REQUEST: %s, RESPONSE: %s" % (word, next)
        self.write(next)

class UpHandler(tornado.web.RequestHandler):
    """
    Handler that takes a comment and outputs the upvote
    """
    def set_default_headers(self):
        #TODO change this to your own domain
        self.set_header("Access-Control-Allow-Origin", "http://localhost:8000")

    def get(self):
        comment = self.get_argument("comment", "")
        comment = clean_comment(comment)
        if comment == "":
            upvote = 0.0
        else:
            upvote = predict_upvotes(comment, UP_DF, UP_FEATURE, UP_REDUCER, UP_LEARNER)
        print "[X] UP REQUEST: %s, UPVOTES: %s" % (comment, str(upvote))
        self.write(str(upvote))

application = tornado.web.Application([
    (r"/sentence", SentenceHandler),
    (r"/up", UpHandler),
])

def init_server(port):
    """
    initializes the server
    """
    print "starting server..."
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("sentence predictor server")
    parser.add_argument("config", help="path to config file", type=str)
    args = vars(parser.parse_args())

    configfile = args["config"]
    print "using config file %s ..." % configfile
    f = open(configfile, "rb")
    config = json.loads(f.read())
    f.close()
    print "configuration:"
    print json.dumps(config, indent = 2)

    print ">>>>>"
    init_sentence(**config["sentence"])
    print ">>>>>"
    init_upvote(**config["upvote"])
    print ">>>>>"
    init_server(**config["server"])
