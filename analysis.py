import json
import pandas as pd
from main import load_subreddit
import sys

def comment_num_words(r):
    return len(r['body'].split(' '))
def comment_length(r):
    return len(r['body'])
def avg_word_length(r):
    words = r['body'].split(' ')
    lengths = [len(word) for word in words]
    return float(sum(lengths)) / len(lengths)
def num_letters(r):
    return sum(1 if c.isalpha() else 0 for c in r['body'])
def num_swear_words(r):
    swears = "fuck shit bitch damn dick crap fag piss pussy asshole slut cock darn douche bastard".split(' ')
    return sum(1 if word in swears else 0 for word in r['body'].split())
# for now, useless, because cleaning step removed capitalization
def capital_letters(r):
    return sum(1 if c.isupper() else 0 for c in r['body'])

if __name__ == "__main__":
    # functions to add to dataframe
    functions = [comment_num_words, comment_length, avg_word_length, num_letters, num_swear_words]

    # pick datafile
    if len(sys.argv) < 3:
        print "Usage: analytics.py datafile outfile"
        sys.exit()

    data_file = sys.argv[1]
    outfile = sys.argv[2]

    df = load_subreddit(data_file)
    for func in functions:
        print "Applying {}...".format(func.__name__)
        df[func.__name__] = df.apply(lambda x: func(x), axis=1)

    # write output
    df.to_csv(outfile, sep='\t', encoding='utf-8')
    print "Wrote to {}".format(outfile)
