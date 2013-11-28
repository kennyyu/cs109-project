import time
import praw
import json

# test file just to try usage of PRAW, real scraper is in scraper.py
# connect to praw, and get the top 200 submissions for this subreddit
print time.time()
r = praw.Reddit(user_agent='User-Agent: awesome comment vote prediction nlp project v1.0 by /u/jkarnage')
submissions = r.get_subreddit('pics').get_top_from_all(limit=1)
submission = next(submissions)
submission.replace_more_comments()
comments = praw.helpers.flatten_tree(submission.comments)
print time.time()

"""
# for each submission get the entire list of comments 
for submission in submissions:
    submission.replace_more_comments()
    comments = praw.helpers.flatten_tree(submission.comments)

    # attributes we want to save for the submission
    sattrs = ['domain', 'ups', 'downs', 'permalink', 'num_comments', 
              'downs', 'over_18', 'stickied', 'selftext', 'title', 'id']

    # attribuets we want to save for the comments, + nested attributes
    cattrs = ['subreddit_id', 'id', 'body', 'parent_id', 'downs', 'ups', 'created']
    cnattrs = [('author', 'name'), ('subreddit', 'display_name')]

    # create the JSON object for the submission, serialize, output to file
    sdict = {}
    for attr in sattrs:
        sdict[attr] = getattr(submission, attr)
    sdict['net'] = sdict['ups'] - sdict['downs']
    print json.dumps(sdict)

    # create the JSON object for each comment, serialize, output to file
    for comment in comments:
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
        print json.dumps(cdict)
""" 
