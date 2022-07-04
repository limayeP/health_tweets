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
    Goal: Read the tweet data
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
    Input(2): file_extension. The default is "*.txt"
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

def get_hashes(w):
    wt_words = []
    for tweet in w.split(' '):
        if tweet.startswith('#'):
            wt_words.append(tweet.strip(','))
    return wt_words

def my_tokenizer(in_string):
    """
    Convert `in_string` of text to a list of tokens using NLTK's TweetTokenizer
    """
    tokenizer = TweetTokenizer(preserve_case=False,
                               reduce_len=True,
                               strip_handles=False)
    tokens = tokenizer.tokenize(in_string)
    return tokens

def organize_dataframe(df):
    """
    Input: dataframe to be organized
    Ouput: dataframe with columns: 'userid', 'date', 'text', 
                                  'news_source', 'hashtags', 'raw_words'
           following was done to the dataframe
          # joining the 16 dataframes by rows
          # initilaize a dataframe list
          # Remove non-ascii characters and the urls
    """
    dflist = []
    # Create a new column
    for k, v in df.items():
        df[k]['news_source'] = k

    for k, v in df.items():
        dflist.append(v)
    # joining the 16 dataframes by rows
    df_all = pd.concat(dflist, axis=0)
    # Remove non-ascii characters and the urls
    df_all["text"] = df_all["text"].apply(remove_non_ascii)
    # Replace blank spaces with NANs and remove rows qith NANs
    df_all['text'].replace('', np.nan, inplace=True)
    df_all.dropna(subset=['text'], inplace=True)
    print(f"The total number of tweets are {len(df_all)}")

    # Collect hashtags into the main dataframe
    df_all["hashtags"] = df_all["text"].apply(lambda x: get_hashes(x))

    # tokenize words without processing
    df_all["raw_words"] = df_all["text"].apply((lambda x: my_tokenizer(x)))
    return df_all

def plot_word_distribution(df_all):
    """
    Input: raw organized dataframe
    Output:  Distribution of top 10 words  without preprocessing
    """
    raw = [val for sublist in df_all["raw_words"] for val in sublist]
    count_raw = Counter(raw).most_common(10)
    df_raw = pd.DataFrame(count_raw, columns=['unprocessed words','frequency'])
    df_raw.plot.bar(x="unprocessed words", y="frequency", rot=0, title="Top trending unprocessed words")
    plt.tight_layout()
    plt.ylabel = ""
    plt.show()

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


def clean_tweets(df_all):
    """
    Input: dataframe to be cleaned
    Output: dataframe where th following was done:
            tokenize and remove stopwords
            keep US and UK and remove 2 letter words
            filter out empty list of words
            change date to datatime format
    """
    # Cleaning tweets
    df_all["filtered_text"] = df_all["text"].apply(lambda x: clean_text(x))
    # Remove rows with empty strings
    df_all['filtered_text'].replace('', np.nan, inplace=True)
    df_all.dropna(subset=['filtered_text'], inplace=True)
    # Tokenize and remove stopwords
    df_all["filtered_words"] = df_all["filtered_text"].apply(lambda x: remove_stopword_sentiment(my_tokenizer(x)))
    # Remove empty list of words
    df_all["filtered_words"] = df_all["filtered_words"].apply(lambda y: np.nan if len(y)==0 else y)
    df_all.dropna(subset=['filtered_words'], inplace=True)
    # Keep US and UK and remove 2 letter words
    df_all["filtered_words"] = df_all["filtered_words"].apply(lambda x: us(x))
    # join filtered words to get cleantext
    df_all["clean_text"] = df_all["filtered_words"].apply(lambda x: " ".join(x))
    # change date to datatime format
    df_all["date"] =df_all["date"].apply(lambda x: pd.to_datetime(x))
    return df_all

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

def tag_lemmatize_tweet_words(df_all):
    """
    IMPORTANT NOTE: nltk.download('omw-1.4')
    Input: cleaned dataframe
    Output: cleaned, tagged and lemmatized dataframe
    """
    # Get tagging and lemmatizing tweet words
    df_all["tagged_words"] = [pos_tag(sent) for sent in df_all['filtered_words']]
    df_all["lemmatized_words"] = df_all["tagged_words"].apply(lambda x: lemmatize_words(x))
    # Remove empty list of words
    df_all["lemmatized_words"] = df_all["lemmatized_words"].apply(lambda y: np.nan if len(y)==0 else y)
    df_all.dropna(subset=['lemmatized_words'], inplace=True)

    # Each tweet cleaned and Lemmatized 
    df_all["lem_clean_text"] = df_all["lemmatized_words"].apply(lambda x: " ".join(x))
    return df_all

def plot_dist_of_processed_words(df_all):
    """
    Input: cleaned, tagged and lemmatized dataframe
    Output: Plot the distribution of cleaned, lemmatized words
    """
    # list of cleaned words with lemmatized all together 
    le_fi = list(df_all["lemmatized_words"])
    le_fil_wrds = [val for sublist in le_fi for val in sublist]

    le_fil =  Counter(le_fil_wrds)
    le_fil_count = Counter(le_fil_wrds).most_common(15)

    le_df_fil = pd.DataFrame(le_fil_count, columns=['processed words','frequency'])
    le_df_fil.plot.bar(x="processed words", y="frequency", rot=70, title="Top trending words")
    plt.tight_layout()
    plt.show()


def plot_tweets_by_year(df_all):
    """
    Input: cleaned, tagged and lemmatized dataframe
    Output: Plot the distribution of cleaned, lemmatized words
    """
    y = df_all["date"].dt.year
    z = df_all["lem_clean_text"].groupby(y).count()
    uy = list(y.unique())
    plt.bar(uy, z)
    plt.xlabel("year")
    # plt.ylabel("Number of tweets")
    plt.title("Number of tweets by year")
    plt.show()

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

def get_raw_tweets_hastags_handles(df_all):
    """
    Input: cleaned, tagged and lemmatized dataframe
    Output: subsetted dataframe with hashtags_handles
    """
    terms = eda(df_all)
    return terms

def plot_hashtags_piechart(terms):
    """
    Input: subsetted dataframe with hashtags_handles
    Output: Plot the distribution of hashtags
    """
    # Plot hashtags
    df_hashtags = pd.DataFrame({"hashtag": terms["hashed"]})
    df_hashtags['hashtag'].value_counts().head(10).plot(kind='pie',
                                                        autopct='%.1f%%', radius=1.2)
    plt.title('Top Trending Hashtags')
    plt.tight_layout()
    plt.show()
    
def plot_handles_piechart(terms):
    """
    Input: subsetted dataframe with hashtags and handles
    Output: Plot the distribution of handles (also known as mentions)
    """
    # Plot distribution of mentions or handles
    df_handles = pd.DataFrame({"handles": terms["handles"]})
    df_handles['handles'].value_counts().head(10).plot(kind='pie',
                                                        autopct='%.1f%%', radius=1.2)
    plt.title('Top Trending Twitter Handles')
    plt.tight_layout()
    #plt.ylabel('')
    plt.show()

def find_hash(sentence):
    if re.findall(r'#ebola'," ".join(sentence), re.IGNORECASE):
        return True
    else:
        return False

def plot_hashtag_ebola_by_year(df_all):
    """
    Input: subsetted dataframe with hashtags_handles
    Output: Plot the distribution of #ebola
    """
    # Subset data with #Ebola containing hashtags
    mask = df_all["hashtags"].apply(lambda x: find_hash(x))
    mdf = df_all[mask == True]

    # Plot number of tweets containing #Ebola by year
    fig, ax = plt.subplots()
    yd = mdf["date"].dt.year
    zd = mdf['text'].groupby(yd).count()
    uyd = yd.unique()
    ax.bar(uyd, zd)
    ax.set_xticks(uyd)
    plt.xlabel("year")
    # plt.ylabel("Number of Ebola tweets")
    plt.title("Number of Ebola tweets by year")
    plt.show()

def dist_tweets_by_news_sources(df_all):
    """
    Input: subsetted dataframe with hashtags_handles
    Output: Plot of the distribution of tweets from new sources
    """
    nz = df_all['lem_clean_text'].groupby(df_all["news_source"]).count()
    nz = pd.DataFrame(nz)
    nz.columns = ["scores"]
    nz['percentage'] = (nz["scores"]/nz["scores"].sum()*100).round(1)
    nz = nz.reset_index()
    nz.plot(x= "news_source", y = "percentage", kind='bar',title="Tweet distribution by newspaper", rot=90)
    plt.tight_layout()
    plt.show()

def plot_dist_of_top_ngrams(df_all, n=15):
    """
    Input(1): cleaned, tagged and lemmatized dataframe
    Input(2): the number of top ngrams to be plotted
    Output: Plot the distribution of ngrams
    """
    # list of cleaned words with lemmatized all together 
    le_fi = list(df_all["lemmatized_words"])
    le_fil_wrds = [val for sublist in le_fi for val in sublist]
    # calculate a range of ngrams using some handy functions
    top_grams = Counter(everygrams(le_fil_wrds, min_len=2, max_len=4))
    top_gm = pd.DataFrame(top_grams.most_common(n))
    top_gm.columns = ["n-grams","count"]
    fig = top_gm.plot(kind="bar", x="n-grams", y="count", title=f"Top {n} n-grams in all tweets")
    plt.tight_layout()
    plt.show()

def plot_bigrams(df_all ,n=5):
    """
    Input(1): cleaned, tagged and lemmatized dataframe
    Input(2): the number of top bigrams to be plotted
    Output: Plot the distribution of bigrams
    """
    # list of cleaned words with lemmatized all together 
    le_fi = list(df_all["lemmatized_words"])
    le_fil_wrds = [val for sublist in le_fi for val in sublist]
    # object of type zip of bigrams from lemmatized filtered words
    bg = ngrams(le_fil_wrds, 2)
    # get the frequency of each bigram '
    bigramFreq = Counter(bg)
     # what are the ten most popular bigrams '
    l = bigramFreq.most_common(n)
    lBg = pd.DataFrame(l, columns=[f"top{n}bigrams", "frequency"])
    lBg.plot.bar(x=f"top{n}bigrams", y="frequency", rot=45, title="Top trending words")
    plt.tight_layout()
    plt.show()

def plot_trigrams(df_all ,n=5):
    """
    Input(1): cleaned, tagged and lemmatized dataframe
    Input(2): the number of top trigrams to be plotted
    Output: Plot the distribution of trigrams
    """
    # list of cleaned words with lemmatized all together 
    le_fi = list(df_all["lemmatized_words"])
    le_fil_wrds = [val for sublist in le_fi for val in sublist]
    # object of type zip of bigrams from lemmatized filtered words
    tg = ngrams(le_fil_wrds, 3)
    # get the frequency of each bigram '
    trigramFreq = Counter(tg)
     # what are the ten most popular bigrams '
    l = trigramFreq.most_common(n)
    lBg = pd.DataFrame(l, columns=[f"top{n}trigrams", "frequency"])
    lBg.plot.bar(x=f"top{n}trigrams", y="frequency", rot=45, title="Top trending words")
    plt.tight_layout()
    plt.show()

def visualize_networks_of_top_ngrams(df_all, n):
    """
    Goal: Visualize networks of top grams
    Input(1): cleaned, tagged and lemmatized dataframe
    Input(2): the number of top ngrams whose connections have to be visualized
    Output: plot of the networks of the top ngrams
    """
    # list of cleaned words with lemmatized all together 
    le_fi = list(df_all["lemmatized_words"])
    le_fil_wrds = [val for sublist in le_fi for val in sublist]
    # calculate a range of ngrams using some handy functions
    top_grams = Counter(everygrams(le_fil_wrds, min_len=2, max_len=4))
    top_gm = pd.DataFrame(top_grams.most_common(n))
    top_gm.columns = ["n-grams","count"]
  
    # Create dictionary of top grams and their counts
    d = top_gm.set_index('n-grams').T.to_dict('records')

    # Create network plot 
    G = nx.Graph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    G.add_node("ebola", weight=100)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title('Networks of top 15 ngrams in News tweets"')
    pos = nx.spring_layout(G, k=2)
    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=16,
                     width=3,
                     edge_color='grey',
                     node_color='lightgreen',
                     with_labels = False,
                     ax=ax
                     )
    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.045
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='white', alpha=0.25),
                horizontalalignment='center', fontsize=12)
    plt.show()

def sentiment_analysis(df_all):
    """
    Input: dataframe
    Output: dataframe with sentiment analysis column
    """
    #Sentiment Analysis
    df_all['scores'] = df_all['lem_clean_text'].apply(lambda Description: sid.polarity_scores(Description))
    df_all.head()

    df_all['compound'] = df_all['scores'].apply(lambda score_dict: score_dict['compound'])
    df_all['sentiment_type']=''
    df_all.loc[df_all.compound>0,'sentiment_type']='POSITIVE'
    df_all.loc[df_all.compound==0,'sentiment_type']='NEUTRAL'
    df_all.loc[df_all.compound<0,'sentiment_type']='NEGATIVE'
    return df_all


def plot_tweet_sentiment_percent(df_all):
    """
    Input: dataframe with sentiment analysis column
    Output: Plot of distribution of tweet sentiment as percentage
    """
                    
    df_sent_count = pd.DataFrame(df_all.groupby("sentiment_type").count().scores)
    df_sent_count['percentage']= (df_sent_count['scores']/df_sent_count['scores'].sum()*100).round(1)
    df_sent_count = df_sent_count.reset_index()
    df_sent_count.plot(x= "sentiment_type", y = "percentage", kind='bar',title="sentiment analysis", rot=0)
    plt.tight_layout()
    plt.show()

def plot_tweet_sentiment_topics(df_all, list_of_topics):
    """
    Input: dataframe with sentiment analysis column
    Output: Plot of distribution of tweet sentiment about "ebola"as percentage
    """
    for i in list_of_topics:
        mask = df_all['lem_clean_text'].str.contains(i)
        b = df_all[mask][["date", "lem_clean_text", "scores", "sentiment_type"]]
        # Distribution of tweet sentiment as percentage
        bb = pd.DataFrame(b.groupby("sentiment_type").count().scores)
        bb['percentage']= (bb['scores']/bb['scores'].sum()*100).round(1)
        bb = bb.reset_index()
        bb.plot(x= "sentiment_type", y = "percentage", kind='bar',title=f"sentiment analysis of {i} tweets", rot=0)
        plt.tight_layout()
    plt.show()

###########################################################
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
