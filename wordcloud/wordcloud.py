import pandas as pd
import json

FIELDS = ["body"]
stopwords = ["a","about","above","across","after","afterwards","again","against","all","almost","alone","along","already","also","although","always","am","among","amongst","amoungst","amount","an","and","another","any","anyhow","anyone","anything","anyway","anywhere","are","around","as","at","back","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","below","beside","besides","between","beyond","bill","both","bottom","but","by","call","can","cannot","cant","co","computer","con","could","couldnt","cry","de","describe","detail","do","done","down","due","during","each","eg","eight","either","eleven","else","elsewhere","empty","enough","etc","even","ever","every","everyone","everything","everywhere","except","few","fifteen","fify","fill","find","fire","first","five","for","former","formerly","forty","found","four","from","front","full","further","get","give","go","had","has","hasnt","have","he","hence","her","here","hereafter","hereby","herein","hereupon","hers","herself","him","himself","his","how","however","hundred","i","ie","if","in","inc","indeed","interest","into","is","it","its","itself","keep","last","latter","latterly","least","less","ltd","made","many","may","me","meanwhile","might","mill","mine","more","moreover","most","mostly","move","much","must","my","myself","name","namely","neither","never","nevertheless","next","nine","no","nobody","none","noone","nor","not","nothing","now","nowhere","of","off","often","on","once","one","only","onto","or","other","others","otherwise","our","ours","ourselves","out","over","own","part","per","perhaps","please","put","rather","re","same","see","seem","seemed","seeming","seems","serious","several","she","should","show","side","since","sincere","six","sixty","so","some","somehow","someone","something","sometime","sometimes","somewhere","still","such","system","take","ten","than","that","the","their","them","themselves","then","thence","there","thereafter","thereby","therefore","therein","thereupon","these","they","thick","thin","third","this","those","though","three","through","throughout","thru","thus","to","together","too","top","toward","towards","twelve","twenty","two","un","under","until","up","upon","us","very","via","was","we","well","were","what","whatever","when","whence","whenever","where","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","whose","why","will","with","within","without","would","yet","you","your","yours","yourself","yourselves", "like", "http", "just", "really", "good", "yeah", "make"]
def load_subreddit(filename, fields=FIELDS):
    """
    Loads the subreddit with the filename and returns
    a dataframe where the column names are the fields
    in the json object.
    """
    file = open(filename, "rb")
    arrays = dict((field, []) for field in fields)
    #arrays = {field:[] for field in fields}
    for line in file.readlines():
        data = json.loads(line)
        for field in fields:
            arrays[field].append(data[field])
    df = pd.DataFrame(arrays)
    file.close()
    return df

def clean_comment(s):
    s = s.lower()
    for c in ',.?;:\'\"[]{}`~!@#$%^&*()=+_\\|':
        s = s.replace(c, '')
    s = s.replace('/', ' ')
    return s

if __name__ == "__main__":
    for subreddit in open("../scraper/scraper/subreddits_important"): 
        subreddit = subreddit.strip('""\n')
        df = load_subreddit('../data/json/' + subreddit)
        
        # write each feature vector out to its assigned file
        d = {}
        for line in df.body:
            words = clean_comment(line).split(' ')
            for word in words:
                if word in d:
                    d[word] += 1
                else:
                    d[word] = 1

        f = open("../data/lines/" + subreddit, "w+")
        for k, v in d.iteritems():
            if v > 100 and k not in stopwords and len(k) > 2:
                for i in xrange(v):
                    f.write((k + ' ').encode("UTF-8"))

        f.close() 
