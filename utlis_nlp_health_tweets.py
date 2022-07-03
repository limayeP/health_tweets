import numpy as np
import pandas as pd
import re
from collections import Counter
import glob
import os
import string
# for vectorization and clustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics

#Dimension reduction
from sklearn.decomposition import PCA

# for cosine calculation
import numpy as np

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import networkx as nx

#nltk
import nltk
from nltk.corpus import stopwords
from nltk.util import everygrams
from nltk.util import ngrams
from nltk.tokenize.casual import TweetTokenizer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
fourgram_measures = nltk.collocations.QuadgramAssocMeasures()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

####################################################################################

def read_tweet(fn, encoding="UTF-8"):
    """
    Goal: REad the tweet data
    """
    file_id = open(fn, "r", encoding=encoding)
    lines = file_id.readlines()
    t = {"userid": [], "date": [], "text": []}
    for ll in lines:
        l = ll.rstrip("\n")
        if l:
            s = l.split("|")
            t["userid"].append(s[0])
            t["date"].append(s[1])
            t["text"].append("".join(s[2:]))
    bf = pd.DataFrame.from_dict(t)
    bf.text = bf.text.str.encode('ascii', 'ignore').str.decode('ascii')
    return bf

def tweet_to_dataframe(datapn, file_extension = "*.txt"):
    """
    Input(1): datapn = "tweets" (folder where tweets are located)
    Input(2): file_extension. The defualt is "*.txt"
    Output: 
    
    """
    source = []
    df = {}

    for fn in glob.glob(os.path.join(datapn, "*.txt")):
        file_name, file_extension = os.path.splitext(fn)
        source = os.path.basename(file_name)
        try:
            df[source] = read_tweet(fn)
        except UnicodeDecodeError:
            df[source] = read_tweet(fn, encoding="ISO-8859-1")
    return df


def remove_non_ascii(tweet):
    tweet = re.sub(r"\\x\d{2}", "", tweet)
    tweet = re.sub(r"http?:\/\/[^\s]*", "", tweet)
    return tweet


def my_tokenizer(in_string):
    """
    Convert `in_string` of text to a list of tokens using NLTK's TweetTokenizer
    """
    tokenizer = TweetTokenizer(preserve_case=False,
                               reduce_len=True,
                               strip_handles=False)
    tokens = tokenizer.tokenize(in_string)
    return tokens


def clean_text(tweet):
    # To remove
    tweet = re.sub(r"\b#.*\b", "", tweet) # hashtags
    tweet = re.sub(r"\b&.*\b", "", tweet) # words that start with &
    tweet = re.sub(r"^RT\s+", "", tweet) # words start with RT one or more spaces
    tweet = re.sub(r"@[A-Za-z0–9_]+", "", tweet) # words start with @ followed by letters or numbers or _
    tweet = re.sub(r" +", " ", tweet) # spaces
    tweet = re.sub(r"[0–9]+", "", tweet) # one or more numbers
    tweet = re.sub(r"[^A-Za-z0–9_. ]+","",tweet) # words begining with letters or numbers or _ followed by. and zero or more spaces followed by space.
    tweet = re.sub(r"[:\(\)]", "", tweet) 
    tweet = re.sub(r"\b\w{1}\b","",tweet)# one letter words
    tweet = re.sub( r"Photo.*", "", tweet) # 
    tweet = re.sub("[" + re.escape(string.punctuation) + "]", "", tweet)
    tweet = re.sub(r"[…—\d+]","", tweet)
    tweet = re.sub(r"\w{14,}", "", tweet) # remove words more 14 or more characters long
    tweet = tweet.replace(u'’', u"'")
    tweet = tweet.replace(r"amp", "")
    tweet = tweet.replace(r"htps", "")
    tweet = tweet.replace(r"com", "")
    tweet = tweet.strip()
    return tweet

def remove_stopword(word_tokens):
    stop_words = stopwords.words('english')+['rt','say', 'says', 'may', 'wel', 'wil']
    for word in stop_words:
        return [word for word in word_tokens if word not in stop_words]


def remove_stopword_sentiment(word_tokens):
    stop_words = stopwords.words('english')+['rt', 'via', 'says']
    words_to_keep = ['don', "don’t", 'ain', 'aren', "aren’t", 'couldn',
                     "couldn’t", 'didn', "didn’t", 'doesn', "doesn’t", 'hadn',
                     "hadn’t", 'hasn', "hasn’t", 'haven', "haven’t", 'isn',
                     "isn’t",'mightn', "mightn’t", 'mustn', "mustn’t", 'needn',
                     "needn’t", 'shan', "shan’t", 'no', 'nor', 'not', 'shouldn',
                     "shouldn’t", 'wasn', "wasn’t", 'weren', "weren’t", 'won',
                     "won’t", 'wouldn', "wouldn’t"]
    for word in stop_words:
        if word in words_to_keep:
            stop_words.remove(word)
    return [word for word in word_tokens if word not in stop_words]

def us(s):
    """
    input: s is a list of tokens
    output: remove 2 letter words that are not US, us, or uk or UK
    """

    lst = []
    l = ['us', 'US','uk', 'UK']
    for i in s:
        if i in l:
            lst.append(i)
        else:
            if re.sub(r"\b\w{2}\b","",i, ):
                lst.append(i)
    return lst

def get_hashes(w):
    wt_words = []
    for tweet in w.split(' '):
        if tweet.startswith('#'):
            wt_words.append(tweet.strip(','))
    return wt_words

def get_hashtags_by_list(lst):
    toplist = ['#healthtalk', '#nhs', '#ebola', '#getfit','#latfit', '#obamacare', '#weightloss','#health', '#fitness', '#recipe']
    for l in lst:
        if l in toplist:
            return True
        else:
            return False

        
def find_hash(sentence):
    if re.findall(r'#ebola'," ".join(sentence), re.IGNORECASE):
        return True
    else:
        return False

def eda(df):
    terms = {"raw": [], "hashed": [],"handles":[]}
    text_lst = list(df["text"])
    wt = " ".join(text_lst)
    terms["raw"].append(my_tokenizer(wt))
    flattened = [val for sublist in terms["raw"] for val in sublist]
    for w in flattened:
        # Count hashtags only
        if re.findall( r'^#.+$', w):
            terms["hashed"].append(re.sub(r":", "", w.lower()))
        # Count handles only
        if re.findall( r'^@.+', w):
            terms["handles"].append(re.sub(r":", "", w.lower()))
    return terms


def get_top_kmeans_words(n_terms, X, clusters, vec):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vec.get_feature_names() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of t


def word_word_freq_lists(wordlist, n):
    count_cbc = Counter([item for item in wordlist])
    tmc_cbc = count_cbc.most_common(n)
    tmc_cbc_words = []
    tmc_cbc_word_frequency = []
    for tmc, freq in tmc_cbc:
        tmc_cbc_words.append(tmc)
        tmc_cbc_word_frequency.append(freq)
    return tmc_cbc_words, tmc_cbc_word_frequency


def lemmatize_words(q):
    lemmatizer = WordNetLemmatizer()
    lms = []
    pos = 'a'
    for s in q:
        if s[1].startswith('NN'):
            pos = 'n'
        elif s[1].startswith('VB'):
            pos = 'v'
        lms.append(lemmatizer.lemmatize(s[0], pos))
    return lms


def build_tfdif_matrix(ngram_range, user_count, text):
    """
    Input: ngram_range = (1,1),
           user_count = df_all["userid"]
           text = df_all["lem_clean_text"]
    Output: vec= vectorizer
            vec_matrix = tfdif matrix
            features = features
    """
    # Number of users
    unique_user_cnt = len(set(user_count))
    
    if ngram_range == (1,1):
        vec= TfidfVectorizer(ngram_range= (1,1), max_features=unique_user_cnt//100,)
    elif ngram_range == (2,2):
        vec= TfidfVectorizer(ngram_range= (2,2), max_features=unique_user_cnt//100,)
    elif ngram_range == (3,3):
        vec= TfidfVectorizer(ngram_range= (3,3), max_features=unique_user_cnt//100,)
    else:
        pass
    print(f"ngram_range is {ngram_range}")
    vec_matrix = vec.fit_transform(text)
    # get feature names
    features = (vec.get_feature_names())
    return vec, vec_matrix, features


def top_ranking_features(tfdif_matrix,features):
    """
    Output: list of top 10 ranking features
    
    """
    sums_vec = tfdif_matrix.sum(axis = 0)
    data = []
    for col, term in enumerate(features):
        data.append( (term, sums_vec[0,col] ))
    ranking = pd.DataFrame(data, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    print ("\n\nWords head : \n", words.head(10))
    return data

def pca_2_components(tfdif_matrix):
    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(tfdif_matrix.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    return x0, x1

def tfidf(lsts):
    """
    input: lsts = text lists

    """
    tfidf = TfidfVectorizer()
    M = tfidf.fit_transform(lsts)
    MT = M.todense().transpose()
    csm = np.matmul(M.todense(), MT) # Cosine Similarity Matrix
    csm = csm.round(decimals=4)
    return csm, tfidf

def max_cosine_similarity(csm, all_news):
    """
    # Input all_news  = list of news agencies
    """
    maxcosine = []
    # Find the max cosine value other than 1
    ii = np.nonzero(csm == csm[csm < 1].max())[0]
    for i in ii:
       n =  all_news[i]
       maxcosine.append(n)
    return maxcosine


# Extracting the texts with certain words
def word_in_text(word, text):
    try:
        text = text.lower()
        match = re,search(word, text)
        if match:
            return True
        return False
    except AttributeError:
        return False
    
def km_tuning(tfdif_matrix, filename):
    # Kmeans clustering
    # Choosing the optimal number of clusters
    # compare a broad range of ks to start
    ks = [2, 50, 200, 500]
    # track a couple of metrics
    sil_scores = []
    inertias = []

    # fit the models, save the evaluation metrics from each run
    for k in ks:
        print('fitting model for {} clusters'.format(k))
        model= KMeans(n_clusters=k,random_state=0)
        model.fit(tfdif_matrix)
        labels = model.labels_
        sil_scores.append(silhouette_score(tfdif_matrix, labels))
        inertias.append(model.inertia_)

    # plot the quality metrics for inspection
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.subplot(211)
    plt.plot(ks, inertias, 'o--')
    #plt.ylabel('inertia')
    plt.title('kmeans parameter search')
    plt.subplot(212)
    plt.plot(ks, sil_scores, 'o--')
    # plt.ylabel('silhouette score')
    plt.xlabel('k');
    plt.savefig(filename)
    plt.close()
    return model

def kmeans_build(best_k, tfdif_matrix, filename):
    """
    Inputs:best_k = best cluster number
           tfdif_matrix = tfdif matrix
           filename = filepath+filename to save the model
    Outputs: KMeans model, save model

    """
    if filename is None:
        filename = '/home/plimaye/Documents/python/twitter_project/collecting_models/kmeans.pkl'
    try:
        kmeans = joblib.load(filename)
        logging.warning("loading cached kmeans clustering model")
    except FileNotFoundError:
        logging.warning("fitting kmeans model")
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        kmeans.fit(tfdif_matrix.todense())
        joblib.dump(kmeans, filename)
    return kmeans

from sklearn import cluster
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
 
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for given clusters
 
    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn
 
    X     :  multidimension np array of data points
 
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape
 
    #compute variance for all clusters beforehand
    cl_var =  (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d+1)
 
    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
 
    return(BIC)


def cluster_freq(df_all, cluster_labels, k, filename):
    """
     Input: df_all = main dataframe containing cleaned tweets
            cluster_labels from the model
            filename = filepath+name to be saved
    """
    # Save cluster labels in the original dataframe
    df_all["clusters"] = cluster_labels 
    # place cluster belonging in the dataframe to plot with news source
    pcount= Counter(df_all["clusters"]).most_common(10)
    df_pcount = pd.DataFrame(pcount, columns=['cluster group','frequency'])
    df_pcount.plot.bar(x="cluster group",
                       y="frequency", rot=0, title="Top 10 tweetclusters")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_kmeans200_news_agencies(df_all, clusters, filename):
    """
    Input: df_all = main dataframe
           clusters= cluster labels from the km model200
           filename = filepath+filename to save the file
    Output: plot of cluster assignment in fraction of the tweets

    """
    fig, ax = plt.subplots(figsize=(11, 8))
    df_all["clusters"] = cluster_labels 
    dfn = pd.DataFrame(df_all, columns=["news_source","clusters"])
    ix = -1
    nsl = unique(dfn['news_source'])
    for ns in nsl:
        x = dfn[dfn['news_source'] == ns]['clusters'].values
        a = np.histogram(x, 200)
        ix += 1
        plt.plot(a[0] / sum(a[0]) + ix*0.2, label=ns)
        plt.grid()
    plt.yticks(np.linspace(0, len(nsl) * 0.2, len(nsl) + 1))
    y = [0.2] * len(nsl)
    y.insert(0, 0)
    ax.set_yticklabels(y)
    plt.grid()
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.xlabel("cluster assignment")
    plt.ylabel("Fraction of news tweets")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def cluster_text_association(type_of_model, model,vec,features, tfdif_matrix, topk=10):
    """
    Input :type_of_model: "KMeans", "HDBSCAN"
           model: sklearn model variable name
           vectorizer: sklearn vectorizer
           features of the matrix obtained with vec.get_feature_names()
           tfidf_matrix: vec_matrix: 
           topk: k numbers of words to get per cluster
    cluster_text_association(type_of_model="KMeans", model=kmeans_4, features=vec[0].get_feature_names(), vec=vec[0], tfdif_matrix=vec_matrix[0], topk=50)
       
    """
    if type_of_model == "KMeans":
        relevant_labels = list(set(model.labels_))
        centroids = model.cluster_centers_.argsort()[:,::-1]
        for this_label in relevant_labels:
            print('Cluster {}:'.format(this_label), end='')
            for ind in centroids[this_label, :topk]:
                print(' {}'.format(features[ind]), end='')
            print()
    elif type_of_model == 'HDBSCAN':
        # ignore noise labels
        relevant_labels = [ x for x in set(model.labels_) if x >= 0 ]
        for this_label in relevant_labels:
            matching_rows = np.where(hdbs.labels_ == this_label)[0]
            coeff_sums = np.sum(tfdif_matrix[matching_rows], axis=0).A1
            sorted_coeff_idxs = np.argsort(coeff_sums)[::-1]
            print('Cluster {}: '.format(this_label), end='')
            for idx in sorted_coeff_idxs[:topk]:
                print('{} '.format(features[idx]), end='')
            print()

def cluster_sample(orig_text, model, idx, preview=15):
    """
    Helper function to display original text for
    those users modeled in cluster `idx`.
    """
    for i,idx in enumerate(np.where(model.labels_ == idx)[0]):
        print(orig_text[idx].replace('\n',' '))
        print()
        if i > preview:
            print('( >>> truncated preview <<< )')
            break
     
pd.set_option('display.max_rows', None)
