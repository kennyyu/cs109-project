"""
misc.py

Additional helper functions needed.
"""

import json

def get_max_ups(pathto_cleaned):
    # need to open subreddits_important file
    namesfile = open('scraper/scraper/subreddits_important', 'r')

    results = {}

    for sr_name in namesfile:
        # need to open the subreddit datafile - assuming in separate dir
        sr_file = open(pathto_cleaned + sr_name[1:-2], 'r')

        max_ups = -1
        sr_id = -1

        for row in sr_file:
            obj = json.loads(row)

            if obj['ups'] > max_ups:
                # replace
                max_ups = obj['ups']

        # debug: print sr_name[1:-2] + ': ' + str(max_ups)
        results[sr_name[1:-2]] = max_ups

        sr_file.close()

    namesfile.close()
    print results
    return results

get_max_ups('../Cleaned/')
