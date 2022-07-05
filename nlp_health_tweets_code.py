 #!/usr/bin/python
from utlis_nlp_health_tweets import *

# Reading tweets into a Pandas dataframe
raw_df = tweet_to_dataframe(datapn ="tweets" , file_extension = "*.txt")

# organize the raw dataframe
org_df = organize_dataframe(raw_df)

# Distribution of top 10 words  without preprocessin
plot_word_distribution(org_df)

# Clean the dataframe
df_clean = clean_tweets(org_df)

# Tag and Lemmatize the tweet words in the dataframe
df = tag_lemmatize_tweet_words(df_clean)

#  Plot the distribution of cleaned, lemmatized words
plot_dist_of_processed_words(df)

# Plot number of tweets by year
plot_tweets_by_year(df)

# Dict with raw tweets, hashtags and handles
dict_hastags_handles = get_raw_tweets_hastags_handles(df)

# Plot the distribution of hashtags as a piechart
plot_hashtags_piechart(dict_hastags_handles)

# Plot the distribution of handles or mentions as a piechart
plot_handles_piechart(dict_hastags_handles)

# Plot the distribution of #ebola
plot_hashtag_ebola_by_year(df)

# distribution of tweets from new sources
dist_tweets_by_news_sources(df)

#  Plot the distribution of ngrams
plot_dist_of_top_ngrams(df, n=20)

# Plot the distribution of bigrams
plot_bigrams(df, n=5)

# Plot the distribution of trigrams
plot_trigrams(df, n=5)

# Visualize networks of top grams
visualize_networks_of_top_ngrams(df, 15)

# Sentiment Analysis
df_sent = sentiment_analysis(df)

# Distribution of tweet sentiment as percentage
plot_tweet_sentiment_percent(df_sent) 

# To inspect a bit more about sentiment of tweets about "ebola"
list_of_topics = ["ebola", "ebola outbreak", "ebola patient"]
plot_tweet_sentiment_topics(df_sent, list_of_topics)

# dataframe of tweets with hashtags 
df_h = dataframe_hashtags(df)

# Cosine similarity matrix of all the lemmatized_filtered tweets gropued by news source
# Plot of the Cosine similarity matrix
dict_tfidf = relation_new_agencies(df)

# Maximum Cosine similarity
news_source = []
for y in df['news_source'].unique():
        news_source.append(y)
max_cosine = max_cosine_similarity(dict_tfidf["lem_csm"], news_source)

# n most common words between the two most similar word lists
msilimar = find_similar(df, max_cosine, n=20)

# To find list of similar pairs of documents
l = top_n_similarity(df, dict_tfidf["lem_csm"], n=5)

# Cosine similarity of the sel_hash subset
#(subset was created based on top 10 hashtags(listed in the variable list_hahtags)
csm, tfidf  = tfidf(df_h['lem_clean_text'])
am = list(sel_hash['lem_clean_text'])
max_csm = max_cosine_similarity(csm, am)

# Plot of distribution of sentiments by hashtags of interest.
list_hashtags = ['^#health\\b', '^#healthtalk\\b', '^#weightloss\\b',
             '^#nhs\\b', '^#ebola\\b', '^#getfit\\b' , '^#latfit\\b' ,
             '^obamacare\\b', '^#fitness\\b', '^#receipe\\b']

plot_sentiments_by_hashtags(sel_hash, list_hashtags)

# To get the tfdif matrix and features of the matrix
# To plot the KMeans clustering results 
vec_matrix, features = tfdif_matrix_with_ngram_hashtags(df_h) 
###################################################################################
# To further investigate clustering of the tweets into 5 clusters
n_clst = 5
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

           
plt.bar(range(len(set(labels))), np.bincount(labels))
plt.ylabel('population')
plt.xlabel('cluster label')
plt.title('cluster populations')
plt.show()


#####################################################################
##############################################################
 # Strongest features in the KMeans model
relevant_labels = list(set(kmeans.labels_))
centroids = kmeans.cluster_centers_.argsort()[:,::-1]
for this_label in relevant_labels:
        print(f"Cluster {this_label}")
        for ind in centroids[this_label, :15]:
                print(f"{features[0][ind]}")

cluster = []
for i in range(0, 5):
        cluster.append(sel_hash[sel_hash["labels"] == i])
        # Subset the dataframe by cluster (one with highest assignment
        clstbigrms = cluster[i]["most_common10_bigrms"].tolist()
        cbflat = [val for sublist in clstbigrms for val in sublist]
        cflat = Counter(cbflat).most_common(10)
        cflatpd= pd.DataFrame(cflat, columns=['top10bigrams','frequency'])
        cflatpd.plot.bar(x="top10bigrams", y="frequency", rot=70, title="Top trending words")
        plt.tight_layout()
        plt.show()
        
        # Distribution of sentiments
        s = cluster[i]["sentiment_type"]
        percent = s.value_counts(normalize=True)
        percent100 = s.value_counts(normalize=True).mul(100).round(1)
        percent100_clust = percent100.rename_axis('sentiments').reset_index(name='percentage')
        percent100_clust.plot(x= "sentiments", y = "percentage", kind='bar',title=f"sentiment analysis of cluster {i} tweets", rot=0)
        plt.tight_layout()
        plt.show()
