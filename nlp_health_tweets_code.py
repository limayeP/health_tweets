datapn = "tweets"

source = []
df = {}

for fn in glob.glob(os.path.join(datapn, "*.txt")):
    file_name, file_extension = os.path.splitext(fn)
    source = os.path.basename(file_name)
    try:
        df[source] = read_tweet(fn)
    except UnicodeDecodeError:
        df[source] = read_tweet(fn, encoding="ISO-8859-1")

# joining the 16 dataframes by rows
# initilaize a dataframe list
dflist = []
# Create a new column
for k, v in df.items():
    df[k]['news_source'] = k
    
for k, v in df.items():
    dflist.append(v)

df_all = pd.concat(dflist, axis=0)

# Remove non-ascii characters and the urls
df_all["text"] = df_all["text"].apply(remove_non_ascii)
print(f"The total number of tweets are {len(df_all)}")
##############################################################
# # of instances stated on website= 58000
# # of rows in dataframe (tweets) = 63326
# checking for reason for discrepancy
len(df_all["userid"].unique())
df_all['text'].replace('', np.nan, inplace=True)
df_all.dropna(subset=['text'], inplace=True)
len(df_all)
# 63325
##############################################################
# Collect hashtags into the main dataframe
df_all["hashtags"] = df_all["text"].apply(lambda x: get_hashes(x))

# tokenize words without processing
df_all["raw_words"] = df_all["text"].apply((lambda x: my_tokenizer(x)))

# Check for any empty lists
# df_all["raw_words"] = df_all["raw_words"].apply(lambda y: np.nan if len(y)==0 else y)
# len(df_all[df_all["raw_words"] =="NAN"])

# Distribution of top 10 words  without preprocessing
raw = [val for sublist in df_all["raw_words"] for val in sublist]
count_raw = Counter(raw).most_common(10)
df_raw = pd.DataFrame(count_raw, columns=['unprocessed words','frequency'])
df_raw.plot.bar(x="unprocessed words", y="frequency", rot=0, title="Top trending unprocessed words")
plt.tight_layout()
plt.ylabel = ""
plt.show()
##############################################################################
# Cleaning tweets
df_all["filtered_text"] = df_all["text"].apply(lambda x: clean_text(x))
# Remove rows with empty strings
df_all['filtered_text'].replace('', np.nan, inplace=True)
df_all.dropna(subset=['filtered_text'], inplace=True)
len(df_all)
# 63192

# Tokenize and remove stopwords
df_all["filtered_words"] = df_all["filtered_text"].apply(lambda x: remove_stopword_sentiment(my_tokenizer(x)))
# Remove empty list of words
df_all["filtered_words"] = df_all["filtered_words"].apply(lambda y: np.nan if len(y)==0 else y)
df_all.dropna(subset=['filtered_words'], inplace=True)
len(df_all)
# 63173

# Keep US and UK and remove 2 letter words
df_all["filtered_words"] = df_all["filtered_words"].apply(lambda x: us(x))

# join filtered words to get cleantext
df_all["clean_text"] = df_all["filtered_words"].apply(lambda x: " ".join(x))

# change date to datatime format
df_all["date"] =df_all["date"].apply(lambda x: pd.to_datetime(x))

# Get tagging and lemmatizing tweet words
df_all["tagged_words"] = [pos_tag(sent) for sent in df_all['filtered_words']]
df_all["lemmatized_words"] = df_all["tagged_words"].apply(lambda x: lemmatize_words(x))
# Remove empty list of words
df_all["lemmatized_words"] = df_all["lemmatized_words"].apply(lambda y: np.nan if len(y)==0 else y)
df_all.dropna(subset=['lemmatized_words'], inplace=True)
len(df_all)
# 63170

# Each tweet cleaned and Lemmatized 
df_all["lem_clean_text"] = df_all["lemmatized_words"].apply(lambda x: " ".join(x))
#########################################################################################################
# dataframe of tweets key = news source
df_sliced_dict = {}
for y in df_all['news_source'].unique():
    df_sliced_dict[y] = df_all[  df_all['news_source'] == y ]

# list of cleaned words without lemmatized all together 
fi = list(df_all["filtered_words"])
fil_wrds = [val for sublist in fi for val in sublist]

# list of cleaned words with lemmatized all together 
le_fi = list(df_all["lemmatized_words"])
le_fil_wrds = [val for sublist in fi for val in sublist]

sep_docs = []
for k, v in df_sliced_dict.items():
    sep_docs.append(" ".join(v["lem_clean_text"]))

# List of news agencies
news_source = []
for y in df_all['news_source'].unique():
    news_source.append(y)
    
# set an index
df_all["ind"] =  list(range(len(df_all["userid"])))
df_all.set_index("ind")


# Distribution of  cleaned words with lemmatized all together 
###########################################################################
le_fil =  Counter(le_fil_wrds)
le_fil_count = Counter(le_fil_wrds).most_common(15)

le_df_fil = pd.DataFrame(le_fil_count, columns=['processed words','frequency'])
le_df_fil.plot.bar(x="processed words", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Plot number of tweets by year
y = df_all["date"].dt.year
z = df_all["lem_clean_text"].groupby(y).count()
uy = list(y.unique())
plt.bar(uy, z)
plt.xlabel("year")
# plt.ylabel("Number of tweets")
plt.title("Number of tweets by year")
plt.show()

terms = eda(df_all)
# w = len([val for sublist in terms["raw"] for val in sublist])
# print(f"The total number of raw words are {w}")
h = len(terms["hashed"])
print(f"The total number of hashes are {h}")
hd = len(terms["handles"])
print(f"The total number of mentions are {hd}")

# Plot hashtags
df_hashtags = pd.DataFrame({"hashtag": terms["hashed"]})
df_hashtags['hashtag'].value_counts().head(10).plot(kind='pie',
                                                    autopct='%.1f%%', radius=1.2)
plt.title('Top Trending Hashtags')
#plt.ylabel('')
plt.tight_layout()
plt.show()

# Subset data with #Ebola containing hashtags
mask = df_all["hashtags"].apply(lambda x: find_hash(x))
mdf = df_all[mask == True]

# Plot number of tweets containing Ebola by year
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

# Plot distribution of mentions
df_handles = pd.DataFrame({"handles": terms["handles"]})
df_handles['handles'].value_counts().head(10).plot(kind='pie',
                                                    autopct='%.1f%%', radius=1.2)
plt.title('Top Trending Twitter Handles')
plt.tight_layout()
#plt.ylabel('')
plt.show()

# distribution of tweets from new sources
nz = df_all['lem_clean_text'].groupby(df_all["news_source"]).count()
nz = pd.DataFrame(nz)
nz.columns = ["scores"]
nz['percentage'] = (nz["scores"]/nz["scores"].sum()*100).round(1)
nz = nz.reset_index()
nz.plot(x= "news_source", y = "percentage", kind='bar',title="Tweet distribution by newspaper", rot=90)
plt.tight_layout()
plt.show()

# calculate a range of ngrams using some handy functions
top_grams = Counter(everygrams(le_fil_wrds, min_len=2, max_len=4))
top_gm = pd.DataFrame(top_grams.most_common(15))
top_gm.columns = ["n-grams","count"]
fig = top_gm.plot(kind="bar", x="n-grams", y="count", title=f"Top 15 n-grams in all tweets")
plt.tight_layout()
plt.show()

# Calculate a range of bigrams from lemmatized filtered words
Bg = ngrams(le_fil_wrds, 2)
Bg

'get the frequency of each bigram '
BigramFreq = Counter(Bg)
BigramFreq

' what are the ten most popular bigrams '
l = BigramFreq.most_common(10)
l
lBg = pd.DataFrame(l, columns=["top10bigrams", "frequency"])
lBg.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Calculate a range of trigrams from lemmatized filtered words
tg = ngrams(le_fil_wrds, 3)
tg

'get the frequency of each bigram '
tigramFreq = Counter(tg)
tigramFreq

' what are the ten most popular bigrams '
l = tigramFreq.most_common(20)
l
lBg = pd.DataFrame(l, columns=["top10bigrams", "frequency"])
lBg.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()


# Visualize Networks of top grams
# Create dictionary oftop grams and their counts
top_gm = pd.DataFrame(top_grams.most_common(15),
                      columns=['ngram', 'count'])
d = top_gm.set_index('ngram').T.to_dict('records')

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
######################################################################################################
#Sentiment Analysis
df_all['scores'] = df_all['lem_clean_text'].apply(lambda Description: sid.polarity_scores(Description))
df_all.head()

df_all['compound'] = df_all['scores'].apply(lambda score_dict: score_dict['compound'])
df_all['sentiment_type']=''
df_all.loc[df_all.compound>0,'sentiment_type']='POSITIVE'
df_all.loc[df_all.compound==0,'sentiment_type']='NEUTRAL'
df_all.loc[df_all.compound<0,'sentiment_type']='NEGATIVE'


# Distribution of tweet sentiment as percentage
df_sent_count = pd.DataFrame(df_all.groupby("sentiment_type").count().scores)
df_sent_count['percentage']= (df_sent_count['scores']/df_sent_count['scores'].sum()*100).round(1)
df_sent_count = df_sent_count.reset_index()
df_sent_count.plot(x= "sentiment_type", y = "percentage", kind='bar',title="sentiment analysis", rot=0)
plt.tight_layout()
plt.show()

    
# To inspect a bit more about sentiment of tweets about a particular topic
# filter on one of those terms to see what the original content was about.
# create a filter series matching "ebola"

mask = df_all['lem_clean_text'].str.contains("ebola")
b = df_all[mask][["date", "lem_clean_text", "scores", "sentiment_type"]]
# Distribution of tweet sentiment as percentage
bb = pd.DataFrame(b.groupby("sentiment_type").count().scores)
bb['percentage']= (bb['scores']/bb['scores'].sum()*100).round(1)
bb = bb.reset_index()
bb.plot(x= "sentiment_type", y = "percentage", kind='bar',title="sentiment analysis of Ebola Tweets", rot=0)
plt.tight_layout()
plt.show()


# Ebola outbreak
mask1 = b['lem_clean_text'].str.contains("ebola outbreak")
b1 = b[mask1][["date", "lem_clean_text", "scores", "sentiment_type"]]
# Distribution of tweet sentiment as percentage
bc = pd.DataFrame(b1.groupby("sentiment_type").count().scores)
bc['percentage']= (bc['scores']/bc['scores'].sum()*100).round(1)
bc = bc.reset_index()
bc.plot(x= "sentiment_type", y = "percentage", kind='bar',title="sentiment analysis of Ebola outbreak Tweets", rot=0)
plt.tight_layout()
plt.show()

# Ebola patient
mask2 = b['lem_clean_text'].str.contains("ebola patient")
b2 = b[mask2][["date", "lem_clean_text", "scores", "sentiment_type"]]
# Distribution of tweet sentiment as percentage
bd = pd.DataFrame(b2.groupby("sentiment_type").count().scores)
bd['percentage']= (bd['scores']/bd['scores'].sum()*100).round(1)
bd = bd.reset_index()
bd.plot(x= "sentiment_type", y = "percentage", kind='bar',title="sentiment analysis of Ebola patient Tweets", rot=0)
plt.tight_layout()
plt.show()

# https://towardsdatascience.com/the-real-world-as-seen-on-twitter-sentiment-analysis-part-one-5ac2d06b63fb
# create a word frequency dictionary
wordfreq = Counter(le_fil_wrds)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
####################################################################################################
# df of tweets with hashtags
df_hash = df_all[df_all.astype(str)['hashtags'] != '[]']
len(df_hash)
# 11344
mk = df_hash['hashtags'].apply(lambda x: get_hashtags_by_list(x))
sel_hash = df_hash[mk]
len(sel_hash)
# 986

##################################################################################################
# Constructing Tfidf matrix from tweets subsetted by top hashtags
vec = []
vec_matrix = []
f_names = []
trfdata = []
vec_x0 = []
vec_x1 = []
x = [(1, 1), (2, 2), (3, 3)]
for v in x:
    a, b, c = build_tfdif_matrix(ngram_range=v,
                                 user_count=sel_hash["userid"],
                                 text=sel_hash["lem_clean_text"])
    trfdata.append(top_ranking_features(b, c))
    d, e = pca_2_components(b)
    vec.append(a)
    vec_matrix.append(b)
    f_names.append(c)
    vec_x0.append(d)
    vec_x1.append(e)
    
# Tuning parameters for Kmeans
ks = [2,5,10,15,20,25,30,35,40,45,50,55,60]

# track a couple of metrics
sil_scores = []
inertias = []

# fit the models, save the evaluation metrics from each run
for k in ks:
    print('fitting model for {} clusters'.format(k))
    model= KMeans(n_clusters=k,random_state=0)
    model.fit(vec_matrix[2])
    labels = model.labels_
    sil_scores.append(silhouette_score(vec_matrix[2], labels))
    inertias.append(model.inertia_)

# plot the quality metrics for inspection
fig, ax = plt.subplots(2, 1, sharex=True)
plt.subplot(211)
plt.plot(ks, inertias, 'o--')
# plt.ylabel('inertia')
plt.title('kmeans parameter search')
plt.subplot(212)
plt.plot(ks, sil_scores, 'o--')
# plt.ylabel('silhouette score')
plt.xlabel('k');
plt.show()

# run_km_tuning = True
# graphpn = "graphs"
# ix = 0
# if run_km_tuning:
#     for z in zip(vec_matrix, vec_x0, vec_x1):
#         ix += 1
#         km_tuning(z[0], os.path.join(graphpn, "kmeans_tuning_gram{ix}.png"))
#         #dendogram(z[1], z[2], filename=os.path.join(graphpn, "dendogram_{ix}.png"))
# #
###########################################################3
# K-Means Processing
number_of_clusters = [53, 16, 7]

n_clst = 53
kmeans = KMeans(n_clusters=n_clst, init='k-means++', random_state=0, max_iter=100, n_init=10, verbose=True)
print("Clustering sparse data with %s" % kmeans)

kmeans.fit(vec_matrix[0])
# cluster_num is the same as cluster labels
cluster_num = kmeans.predict(vec_matrix[0])

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
labels_unique = np.unique(labels)
 
lenlb = len(labels_unique)
label_elem = np.zeros([lenlb])

#place cluster numbers and cluster labels in the df
# Save cluster labels in the original dataframe
sel_hash["labels"] = labels
sel_hash["Cluster_Num"] = cluster_num

# Top Cluster Pair in each tweet
sel_hash['bigrms'] = sel_hash.apply(lambda sel_hash: BigramCollocationFinder.from_words(sel_hash['lemmatized_words']),axis=1)

sel_hash['most_common10_bigrms'] = sel_hash['bigrms'].apply(lambda sel_hash: sel_hash.nbest(bigram_measures.pmi, 10))

# Group the df by the cluster label or Cluster_num
cluster_top_pair = sel_hash.groupby("Cluster_Num")
elem_cluster = np.bincount(labels) # Number of elements per Cluster
 
for i in labels_unique:
    label_elem[i] = 0
     
    for l in labels:
        if l == i: label_elem[i] +=1
    print("Label = ", i, "  Number of Elements = ", label_elem[i])

num_tweets = len(sel_hash)
samp_size = min(num_tweets, 300) 
 
silh_score = metrics.silhouette_score(vec_matrix[0], labels, metric='euclidean', sample_size=samp_size)
print("Silhouette score = ", round(silh_score, 3), "  for Sample Size = ", samp_size)
 
cluster_arr = vec_matrix[0].todense()
BIC = compute_bic(kmeans,cluster_arr)
print('BIC Score = ', round(BIC, 3))

# set image size
plt.figure(figsize=(12, 7))
sns.scatterplot(x=vec_x0[0], y=vec_x1[0], hue=labels, palette="viridis")
# set a title
plt.title(" Cluster visualization after PCA", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
plt.show()
           
plt.bar(range(len(set(labels))), np.bincount(labels))
plt.ylabel('population')
plt.xlabel('cluster label')
plt.title('cluster populations')
plt.show()

# Cosine similarity of the sel_hash subset (subset was created based on top 10 hashtags)
csm, tfidf  = tfidf(sel_hash['lem_clean_text'])
am = list(sel_hash['lem_clean_text'])
max_csm = max_cosine_similarity(csm, am)
#####################################################################
##############################################################
 # Strongest features in the KMeans model
relevant_labels = list(set(kmeans.labels_))
centroids = kmeans.cluster_centers_.argsort()[:,::-1]
f_names[0]
for this_label in relevant_labels:
            print('Cluster {}:'.format(this_label), end='')
            for ind in centroids[this_label, :15]:
                print(' {}'.format(f_names[0][ind]), end='')
            print()

# Collect the clusters to look at each of them closely
relevant_labels = list(set(kmeans.labels_))
centroids = kmeans.cluster_centers_.argsort()[:,::-1]
f_names[0]
lst = list(sel_hash["lem_clean_text"])
for this_label in relevant_labels:
            print('Cluster {}:'.format(this_label), end='')
            for ind in centroids[this_label, :15]:
                print(' {}'.format(lst[ind]), end='')
            print()

# Subset the dataframe by cluster (one with highest assignment
cluster3 = sel_hash[sel_hash["labels"] == 3]
clst3bigrms = cluster3["most_common10_bigrms"].tolist()
cbflat = [val for sublist in clst3bigrms for val in sublist]
cflat = Counter(cbflat).most_common(10)
cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Distribution of sentiments
s = cluster3["sentiment_type"]
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1)
percent100_clust3 = percent100.rename_axis('sentiments').reset_index(name='percentage')
percent100_clust3.plot(x= "sentiments", y = "percentage", kind='bar',title="sentiment analysis of cluster 3 tweets", rot=0)
plt.tight_layout()
plt.show()
#############################################################################

cluster2 = sel_hash[sel_hash["labels"] == 2]
clst2bigrms = cluster2["most_common10_bigrms"].tolist()
cbflat = [val for sublist in clst2bigrms for val in sublist]
cflat = Counter(cbflat).most_common(10)
cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Distribution of sentiments
s = cluster2["sentiment_type"]
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1)
percent100_clust2 = percent100.rename_axis('sentiments').reset_index(name='percentage')
percent100_clust2.plot(x= "sentiments", y = "percentage", kind='bar',title="sentiment analysis of cluster 2 tweets", rot=0)
plt.tight_layout()
plt.show()
####################################################################################


cluster4 = sel_hash[sel_hash["labels"] == 4]
clst4bigrms = cluster4["most_common10_bigrms"].tolist()
cbflat = [val for sublist in clst4bigrms for val in sublist]
cflat = Counter(cbflat).most_common(10)
cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Distribution of sentiments
s = cluster4["sentiment_type"]
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1)
percent100_clust4 = percent100.rename_axis('sentiments').reset_index(name='percentage')
percent100_clust4.plot(x= "sentiments", y = "percentage", kind='bar',title="sentiment analysis of cluster 4 tweets", rot=0)
plt.tight_layout()
plt.show()

###############################################################################################
# Counts of hastags and labels
h = sel_hash[["hashtags", "labels"]]

# unlist the list in hashtags
h['hashtags'] = h['hashtags'].apply(lambda x: ' '.join(dict.fromkeys(x).keys()))

# groupby hashtags and labels
hc  = (h.groupby(["hashtags", "labels"]).size().reset_index()
               .rename(columns={0 : 'count'}))

# Count of clusters with #health
rslt_df = hc[hc['hashtags'].str.contains(r'^#health\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #Health")
plt.show()

# Count of clusters with #healthtalk
rslt_df = hc[hc['hashtags'].str.contains(r'^#healthtalk\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #Healthtalk")
plt.show()

# Count of clusters with #weightloss
rslt_df = hc[hc['hashtags'].str.contains(r'^#weightloss\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #weightloss ")
plt.show()

# Count of clusters with #nhs
rslt_df = hc[hc['hashtags'].str.contains(r'^#nhs\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #nhs ")
plt.show()

# Count of clusters with #ebola
rslt_df = hc[hc['hashtags'].str.contains(r'^#ebola\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #ebola")
plt.show()

# Count of clusters with #getfit
rslt_df = hc[hc['hashtags'].str.contains(r'^#getfit\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #getfit")
plt.show()

# Count of clusters with #latfit
rslt_df = hc[hc['hashtags'].str.contains(r'^#latfit\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #latfit")
plt.show()

# Count of clusters with #obamacare
rslt_df = hc[hc['hashtags'].str.contains(r'^#obamacare\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #obamacare")
plt.show()

# Count of clusters with #fitness
rslt_df = hc[hc['hashtags'].str.contains(r'^#fitness\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #fitness")
plt.show()

# Count of clusters with #receipe
rslt_df = hc[hc['hashtags'].str.contains(r'^#receipe\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #receipe")
plt.show()

############################################################################################################
# FULL MODEL
##############################################################################################################
##################################################################################################
# Constructing Tfidf matrix from tweets subsetted by top hashtags
vec = []
vec_matrix = []
f_names = []
trfdata = []
vec_x0 = []
vec_x1 = []
x = [(1, 1), (2, 2), (3, 3)]
for v in x:
    a, b, c = build_tfdif_matrix(ngram_range=v,
                                 user_count=df_all["userid"],
                                 text=df_all["lem_clean_text"])
    trfdata.append(top_ranking_features(b, c))
    d, e = pca_2_components(b)
    vec.append(a)
    vec_matrix.append(b)
    f_names.append(c)
    vec_x0.append(d)
    vec_x1.append(e)
    
# Tuning parameters for Kmeans
ks = [2, 50, 200, 500]
# track a couple of metrics
sil_scores = []
inertias = []

igram = 2
# fit the models, save the evaluation metrics from each run
for k in ks:
    print('fitting model for {} clusters'.format(k))
    model= KMeans(n_clusters=k,random_state=0)
    model.fit(vec_matrix[igram])
    labels = model.labels_
    sil_scores.append(silhouette_score(vec_matrix[igram], labels))
    inertias.append(model.inertia_)

# plot the quality metrics for inspection
fig, ax = plt.subplots(2, 1, sharex=True)
plt.subplot(211)
plt.plot(ks, inertias, 'o--')
plt.title('kmeans parameter search')
ax[0].set_ylabel('inertia')
plt.subplot(212)
plt.plot(ks, sil_scores, 'o--')
ax[1].set_ylabel('silhouette score')
ax[1].set_xlabel('k');
plt.show()

# run_km_tuning = True
# graphpn = "graphs"
# ix = 0
# if run_km_tuning:
#     for z in zip(vec_matrix, vec_x0, vec_x1):
#         ix += 1
#         km_tuning(z[0], os.path.join(graphpn, "kmeans_tuning_gram{ix}.png"))
#         #dendogram(z[1], z[2], filename=os.path.join(graphpn, "dendogram_{ix}.png"))
# #
###########################################################3
# K-Means Processing
#number_of_clusters = [53, 16, 7]

n_clst = 200
kmeans = KMeans(n_clusters=n_clst, init='k-means++', random_state=0, max_iter=100, n_init=10, verbose=True)
print("Clustering sparse data with %s" % kmeans)

kmeans.fit(vec_matrix[igram])
# cluster_num is the same as cluster labels
cluster_num = kmeans.predict(vec_matrix[igram])

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
labels_unique = np.unique(labels)
 
lenlb = len(labels_unique)
label_elem = np.zeros([lenlb])

#place cluster numbers and cluster labels in the df
# Save cluster labels in the original dataframe
df_all["labels"] = labels
df_all["Cluster_Num"] = cluster_num

# Top Cluster Pair in each tweet
df_all['bigrms'] = df_all.apply(lambda df_all: BigramCollocationFinder.from_words(df_all['lemmatized_words']),axis=1)

df_all['most_common10_bigrms'] = df_all['bigrms'].apply(lambda df_all: df_all.nbest(bigram_measures.pmi, 10))

# Group the df by the cluster label or Cluster_num
cluster_top_pair = df_all.groupby("Cluster_Num")
elem_cluster = np.bincount(labels) # Number of elements per Cluster
 
for i in labels_unique:
    label_elem[i] = 0
     
    for l in labels:
        if l == i: label_elem[i] +=1
    print("Label = ", i, "  Number of Elements = ", label_elem[i])

num_tweets = len(df_all)
samp_size = min(num_tweets, 300) 
 
silh_score = metrics.silhouette_score(vec_matrix[igram], labels, metric='euclidean', sample_size=samp_size)
print("Silhouette score = ", round(silh_score, 3), "  for Sample Size = ", samp_size)
 
cluster_arr = vec_matrix[igram].todense()
BIC = compute_bic(kmeans,cluster_arr)
print('BIC Score = ', round(BIC, 3))

# set image size
plt.figure(figsize=(12, 7))
sns.scatterplot(x=vec_x0[igram], y=vec_x1[igram], hue=labels, palette="viridis")
# set a title
plt.title(" Cluster K 200, tri-gram visualization after PCA", fontdict={"fontsize": 18})
# set axes names
#plt.ylabel("X1", fontdict={"fontsize": 16})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.show()
           
plt.bar(range(len(set(labels))), np.bincount(labels))
plt.title('k200 trigram cluster distribution')
#plt.ylabel('population')
plt.xlabel('cluster label')
plt.show()

# Cosine similarity of the sel_hash subset (subset was created based on top 10 hashtags)
csm, tfidf  = tfidf(sel_hash['lem_clean_text'])
am = list(sel_hash['lem_clean_text'])
max_csm = max_cosine_similarity(csm, am)
#####################################################################
##############################################################
 # Strongest features in the KMeans model
relevant_labels = list(set(kmeans.labels_))
centroids = kmeans.cluster_centers_.argsort()[:,::-1]
f_names[0]
for this_label in relevant_labels:
            print('Cluster {}:'.format(this_label), end='')
            for ind in centroids[this_label, :15]:
                print(' {}'.format(f_names[0][ind]), end='')
            print()


# Subset the dataframe by cluster (one with highest assignment
cluster34 = df_all[df_all["labels"] == 34]
clst34bigrms = cluster34["most_common10_bigrms"].tolist()
cbflat = [val for sublist in clst34bigrms for val in sublist]
cflat = Counter(cbflat).most_common(10)
cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Distribution of sentiments
s = cluster34["sentiment_type"]
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1)
percent100_clust34 = percent100.rename_axis('sentiments').reset_index(name='percentage')
percent100_clust34.plot(x= "sentiments", y = "percentage", kind='bar',title="sentiment analysis of cluster 
#34 tweets", rot=0)
plt.tight_layout()
plt.show()
#############################################################################
x = np.bincount(labels)
np.sort(x)
np.where(x==np.sort(x)[-2])

cluster94 = df_all[df_all["labels"] == 94]
clst94bigrms = cluster94["most_common10_bigrms"].tolist()
cbflat = [val for sublist in clst94bigrms for val in sublist]
cflat = Counter(cbflat).most_common(10)
cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Distribution of sentiments
s = cluster94["sentiment_type"]
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1)
percent100_clust2 = percent100.rename_axis('sentiments').reset_index(name='percentage')
percent100_clust2.plot(x= "sentiments", y = "percentage", kind='bar',title="sentiment analysis of cluster 2 tweets", rot=0)
plt.tight_layout()
plt.show()
####################################################################################
np.where(x==np.sort(x)[-3])

cluster4 = df_all[df_all["labels"] == 2]
clst4bigrms = cluster4["most_common10_bigrms"].tolist()
cbflat = [val for sublist in clst4bigrms for val in sublist]
cflat = Counter(cbflat).most_common(10)
cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
plt.tight_layout()
plt.show()

# Distribution of sentiments
s = cluster4["sentiment_type"]
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1)
percent100_clust4 = percent100.rename_axis('sentiments').reset_index(name='percentage')
percent100_clust4.plot(x= "sentiments", y = "percentage", kind='bar',title="sentiment analysis of cluster 4 tweets", rot=0)
plt.tight_layout()
plt.show()

###############################################################################################
# Counts of hastags and labels
h = df_all[["hashtags", "labels"]]

# unlist the list in hashtags
h['hashtags'] = h['hashtags'].apply(lambda x: ' '.join(dict.fromkeys(x).keys()))

# groupby hashtags and labels
hc  = (h.groupby(["hashtags", "labels"]).size().reset_index()
               .rename(columns={0 : 'count'}))

# Count of clusters with #health
rslt_df = hc[hc['hashtags'].str.contains(r'^#health\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #Health")
plt.show()

# Count of clusters with #healthtalk
rslt_df = hc[hc['hashtags'].str.contains(r'^#healthtalk\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #Healthtalk")
plt.show()

# Count of clusters with #weightloss
rslt_df = hc[hc['hashtags'].str.contains(r'^#weightloss\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #weightloss ")
plt.show()

# Count of clusters with #nhs
rslt_df = hc[hc['hashtags'].str.contains(r'^#nhs\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #nhs ")
plt.show()

# Count of clusters with #ebola
rslt_df = hc[hc['hashtags'].str.contains(r'^#ebola\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #ebola")
plt.show()

# Count of clusters with #getfit
rslt_df = hc[hc['hashtags'].str.contains(r'^#getfit\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #getfit")
plt.show()

# Count of clusters with #latfit
rslt_df = hc[hc['hashtags'].str.contains(r'^#latfit\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #latfit")
plt.show()

# Count of clusters with #obamacare
rslt_df = hc[hc['hashtags'].str.contains(r'^#obamacare\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #obamacare")
plt.show()

# Count of clusters with #fitness
rslt_df = hc[hc['hashtags'].str.contains(r'^#fitness\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #fitness")
plt.show()

# Count of clusters with #receipe
rslt_df = hc[hc['hashtags'].str.contains(r'^#receipe\b')]
del rslt_df["count"]
ab = rslt_df['labels'].value_counts()
ab.plot(kind='bar')
plt.xticks(rotation=25)
plt.xlabel("Cluster labels")
plt.title("Frequency Distriubtion of tweets with #receipe")
plt.show()

#############################################################################################################
#16 news agencies
# Cosine similarity matrix of all the lemmatized_filtered tweets gropued by news source
lem_clean_csm, tfidf_lem_clean = tfidf(sep_docs)

features_text = tfidf_lem_clean.get_feature_names()
lem_clean_csm.shape
# Look at the cosine matrix
%pylab
plt.rcParams["figure.dpi"] = 500
plt.rc('image', cmap='nipy_spectral')
plt.matshow(lem_clean_csm)
plt.colorbar()

max_cosine = max_cosine_similarity(lem_clean_csm, news_source)
print(f"The most similar news agency tweets are: {max_cosine_similarity(lem_clean_csm, news_source)}")


# Find the common words in two docs that are most similar
word_list_cbc = [item for sublist in df_all[df_all['news_source'] == max_cosine[0]]['filtered_words'] for item in sublist]

word_list_npr = [item for sublist in df_all[df_all['news_source'] == max_cosine[1]]['filtered_words'] for item in sublist]


print(f"The common words between two news agencies that have the closest words in tweets are: {len(set(word_list_cbc)&set(word_list_npr))}")


# Too many words so find the 20 most common words
words_cbc, wordfreq_cbc = word_word_freq_lists(word_list_cbc, 50)
words_npr, wordfreq_npr = word_word_freq_lists(word_list_npr, 50)

print(f'Words common among the 50 most common words in cbc and npr tweets are:{list(set(words_npr).intersection(words_cbc))}')



# The cosine similarity matrix is mirror image  matrix
# To get top five values, 10 will have to be found
lem_clean_csm.shape
flat_csm = csm.flatten()
flat_csm.shape
# sort flat_csm
flat_csm_sorted = np.sort(flat_csm)
s = flat_csm_sorted[flat_csm_sorted != 1]
top_5_values = flipud(s[-10:][::2])
most_similar_docs = {}
for ix, t in enumerate(top_5_values):
    z = np.where(csm == t)[0]
    most_similar_docs[ix] = (df_all[df_all['news_source'][z[0]],  df_all[df_all['news_source'][z[1]])

