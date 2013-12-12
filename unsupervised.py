"""
Unsupervised Learning techniques for reddit
"""
from sklearn.cluster import KMeans, Ward, SpectralClustering
from nltk.corpus import stopwords

def cluster_within_subreddit(df, X, n_clusters):
    predictor = KMeans(n_clusters)
    Y = predictor.fit_predict(X)
    clusters = [[] for _ in range(n_clusters)]
    for i, cluster in enumerate(Y):
        clusters[cluster].append(i)

    # for each cluster, print out out top M words
    frequent = [{} for _ in range(n_clusters)]
    for cluster_num, ixlist in enumerate(clusters):
        for ix in ixlist:
            words = df.ix[ix]["body"].split()
            for word in words:
                if word not in frequent[cluster_num]:
                    frequent[cluster_num][word] = 0
                frequent[cluster_num][word] += 1

    # remove stopwords
    s = set(stopwords.words('english'))
    file = open('data/stopwords.txt', 'rb')
    s |= set(file.readlines())
    file.close()

    M_words_to_print = 30
    for cluster_num, word_freq in enumerate(frequent):
        print ">>> CLUSTER:", cluster_num, "num items:", len(clusters[cluster_num])
        words = filter(lambda (k,v): not k in s, word_freq.items())
        for (word, freq) in sorted(words, reverse=True, key=lambda (k,v): v)[:M_words_to_print]:
            print word, freq
        print ""
