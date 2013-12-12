import argparse
import tornado.ioloop
import tornado.web

import features
from sentence import *
from main import *

NGRAM = 6
DATAFILE = "data/Liberal"
RANDOM = False
NGRAM_FREQ = None

class PredictHandler(tornado.web.RequestHandler):
    def get(self):
        word = self.get_argument("word", "")
        word = clean_comment(word)
        print "REQUEST: %s" % word
        if word == "":
            self.write("")
            return
        next = find_next_word(word, NGRAM_FREQ, random=RANDOM)
        self.write(next)

application = tornado.web.Application([
    (r"/sentence", PredictHandler),
])

def init_sentence(subreddit, ngram, random):
    global DATAFILE, NGRAM, RANDOM, NGRAM_FREQ

    DATAFILE = subreddit
    NGRAM = ngram
    RANDOM = random

    model = features.NGramModel(NGRAM)
    print "N used for ngram: %d" % NGRAM

    df = load_subreddit(DATAFILE)
    print "loaded: %s" % DATAFILE

    # Make the training set
    print "making training data..."
    X_train, Y_train = model.make_training_xy(df)

    # Build most likely sentence with a root word
    NGRAM_FREQ = make_ngram_freq(X_train, model)

def init_server(port):
    print "starting server..."
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

def main(subreddit, ngram, random, port):
    init_sentence(subreddit, ngram, random)
    init_server(port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("sentence predictor server")
    parser.add_argument("subreddit", help="path to subreddit data file", type=str)
    parser.add_argument("ngram", help="ngram to use", type=int)
    parser.add_argument("--random", help="do random", dest="random", action="store_true")
    parser.add_argument("--port", help="port for server", dest="port", type=int, default=8888)
    args = vars(parser.parse_args())
    main(**args)
