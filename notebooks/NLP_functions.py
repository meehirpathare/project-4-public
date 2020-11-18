from datetime import datetime
import json
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as open_url
from NLPPipe import NLPPipe
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from cleaner_funcs import clean, clean_text_string, clean_list
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer
import csv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

def add_stopwords(stop_words, words_to_add):
    """
    Function that adds words to a list or set of stopwords for NLP processing

    Parameters:
    current_set (list or set): Self-explanatory
    words_to_add (list or set): Self-explanatory

    Returns:
    set_of_stopwords (set): New set of stopwords
    """

    # convert to list if given a set to make mutable
    stop_words = list(stop_words)
    words_to_add = list(words_to_add)

    for word in words_to_add:
        stop_words.append(word)

    set_of_stopwords = set(stop_words)
    return set_of_stopwords

def get_kmeans_wcss(my_stop_words, max_num, corpus):
    """
    Function that returns a list of wcss values
    
    Parameters:
    my_stop_words (list or set): stop words
    max_num (int): maximum number of features to fit
    corpus (list): processed text corpus

    Returns:
    nlp (NLPPipe): fit and transformed Tfidf model
    wcss (list): error values
    
    """
    nlp = NLPPipe(vectorizer=TfidfVectorizer(stop_words=set(my_stop_words),max_features=max_num), 
                  cleaning_function=clean, 
                  tokenizer=TreebankWordTokenizer().tokenize, 
                  stemmer=PorterStemmer())

    nlp.fit(corpus)
    vectors = nlp.transform(corpus)

    X_data = vectors.todense()
    print(X_data.shape)

    wcss = []
    for i in range(1,11): 
        kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0)
        kmeans.fit(X_data)

        wcss.append(kmeans.inertia_)
    return nlp, wcss

def get_top_keywords(df, clusters, labels, n_terms):
    """
    Function returns word clusters of length n_terms
    
    Parameters:
    df (pd.DataFrame): dense dataframe of word frequency
    clusters (kmeans.fit_predict): labelled documents
    n_terms (int): number of words in groupings
    
    Returns: prints word clusters
    """
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

def plot_tsne_pca(data, labels, save=False, filepath='', filename='PCA TSNE.png'):
    """
    Function plots PCA and TSNE side-by-side
    
    Parameters: 
    data (array): Dense array of word frequencies
    labels (kmeans.fit_predict): clusters 
    
    Optional args:
    save (binary): self-explanatory
    filepath: self-explanatory
    filename: self-explanatory
    
    Returns: None
    """
    
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:])
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:]))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(22, 10))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot', fontsize=32)

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot', fontsize=32)

    if save == True:
        plt.savefig(filepath + filename)

def top_features(nlp_obj, lower, upper):
    """
     Function that returns ranked topics from an NLPPipe object

    Parameters: 
    nlp_obj (NLPPipe): Object of NLPPipe class. See NLPPipe.py for details
    lower (int):Index of most frequent word to return 
    upper (int):Index of least frequent word to return

    Returns:
    top_features (list):list of words ordered by frequency in given range
    """

    # get tokens sorted by frequency and their indices
    indices = np.argsort(nlp_obj.vectorizer.idf_)
    features = nlp_obj.vectorizer.get_feature_names()
    top_n = upper
    top_features = [features[i] for i in indices[lower:top_n]]
    return top_features