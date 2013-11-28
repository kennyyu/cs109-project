from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from operator import itemgetter
import praw
import itertools
import sys

# attributes we want to save for the submission
sattrs = ['domain', 'ups', 'downs', 'permalink', 'num_comments', 'created', 
          'downs', 'over_18', 'stickied', 'selftext', 'title', 'id']

# attributes we want to save for the comments, + nested attributes
cattrs = ['subreddit_id', 'id', 'body', 'downs', 'ups', 'created']
cnattrs = [('author', 'name'), ('subreddit', 'display_name')]

def get_top_submissions(subreddit, n):
    # connect to praw, and get the top 200 submissions for this subreddit
    r = praw.Reddit(user_agent='User-Agent: awesome comment vote prediction nlp project ' + subreddit + ' v1.0 by /u/jkarnage')
    submissions = r.get_subreddit(subreddit).get_top_from_all(limit=n)
    return submissions

def get_submission_json(submission):
    # create the JSON object for the submission
    sdict = {}
    for attr in sattrs:
        sdict[attr] = getattr(submission, attr)
    sdict['net'] = sdict['ups'] - sdict['downs']
    return sdict

def get_submission_comments(submission):
    submission.replace_more_comments()
    comments = praw.helpers.flatten_tree(submission.comments)
    return comments

def get_comment_json(comment):
    # create the JSON object for the comment
    cdict = {}
    
    # regular attributes, flat access 
    for attr in cattrs:
        cdict[attr] = getattr(comment, attr)

    # attributes where we have to index twice, i.e. comment.subreddit.display_name 
    for attr in cnattrs:
        try:
            cdict[attr[0]] = getattr(getattr(comment, attr[0]), attr[1])
        except:
            cdict[attr[0]] = "[deleted]"

    cdict['net'] = cdict['ups'] - cdict['downs']
    return cdict


class RedditScraper(MRJob):
  INPUT_PROTOCOL  = JSONValueProtocol
  OUTPUT_PROTOCOL = JSONValueProtocol

  def mapper(self, _, subreddit):
    # for each subreddit, run an API call to fetch the top 200 posts
    try:
        submissions = get_top_submissions(subreddit, 200)
        for submission in submissions: 
            # XXX: we don't really need this for now, we can always request them later
            # get the json for the submission, yield for later processing
            # sjson = get_submission_json(submission)
            # yield sjson["id"], sjon
        
            # get json for each of a submission's comments, serialize and send to reducer
            try:
                comments = get_submission_comments(submission)
                for comment in comments:
                    cjson = get_comment_json(comment)
                    cjson["post_id"]      = submission.id
                    cjson["post_created"] = submission.created
                    cjson["post_ups"]     = submission.ups 
                    cjson["post_downs"]   = submission.downs
                    cjson["post_net"]     = submission.ups - submission.downs
                    yield cjson["subreddit"], cjson
            except:
                sys.stderr.write(subreddit + " submission failed\n")
    except:
        sys.stderr.write(subreddit + " failed\n")

    # keep track of the number of subreddits we've processed
    sys.stderr.write(subreddit + " processed\n")
  
  def reducer(self, subreddit, comments):
    # sort the results by timestamp within each subreddit before outputting
    comments = sorted(comments, key=itemgetter('created')) 

    # yield the subreddit name as a demarcator in the final file
    yield "", "<--- SUBREDDIT " + subreddit + " --->"
    
    # yield each of the comments on its own line
    for comment in comments:
      yield "", comment

if __name__ == '__main__':
  RedditScraper.run()
