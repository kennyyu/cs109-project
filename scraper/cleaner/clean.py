from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from operator import itemgetter
import itertools
import nltk

class Subredditize(MRJob):
  INPUT_PROTOCOL  = JSONValueProtocol
  OUTPUT_PROTOCOL = JSONValueProtocol

  def mapper(self, _, comment):
    # remove comments with less than 10 total votes
    # if comment["ups"] + comment["downs"] < 10:
    #  return

    # remove subreddit headers or bogus lines
    if isinstance(comment, str):
      return

    # remove comments whose score is hidden
    #if comment["score_hidden"]:
    #  return
    
    # remove comments that were deleted
    if comment["body"] == "[deleted]":
      return

    # perform cleaning on body text to standard representation
    # treat leading and trailing punctuation as own words, lowercase
    tokens = nltk.word_tokenize(comment["body"])
    tokens[:] = [token.strip('\'!@#$%^&*-+_=|\/"><[](){}?:!.,;').lower() for token in tokens]
    comment["body"] = " ".join(tokens)

    # yield to final file
    yield comment["subreddit"], comment
  
  def reducer(self, subreddit, comments):
    # sort the results by timestamp within each subreddit
    comments = sorted(comments, key=itemgetter('created')) 

    # demarcator for split script
    yield "", "<--- SUBREDDIT " + subreddit + " --->"
    
    # yield each comment in order
    for comment in comments:
      yield "", comment

if __name__ == '__main__':
  Subredditize.run()
