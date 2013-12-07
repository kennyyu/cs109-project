"""
Analyze subreddits

Data format:

{
  "body": string
  "post_ups": int
  "subreddit_id": string
  "created": float (timestamp)
  "downs": int
  "author": string
  "post_net": int
  "subreddit": string
  "post_id": string
  "post_downs": int
  "net": int
  "ups": int
  "id": string
  "post_created": float
}
"""

import json
import pandas as pd
import numpy as np

FIELDS = ["body", "post_ups", "subreddit_id", "created", "downs",
          "author", "post_net", "subreddit", "post_id", "post_downs",
          "net", "ups", "id", "post_created"]


def load_subreddit(filename, fields=FIELDS):
    """
    Loads the subreddit with the filename and returns
    a dataframe where the column names are the fields
    in the json object.
    """
    file = open(filename, "rb")
    arrays = {field:[] for field in fields}
    for line in file.readlines():
        data = json.loads(line)
        for field in fields:
            arrays[field].append(data[field])
    df = pd.DataFrame(arrays)
    file.close()
    return df

DATA_FILE = "data/Liberal.txt"

df = load_subreddit(DATA_FILE)

print df.head(5)


