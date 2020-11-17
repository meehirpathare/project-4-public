import numpy as np
from urllib.request import urlopen as open_url

def top_features(nlp_obj, lower, upper):
    """
     Function that returns ranked topics from an NLPPipe object
        Parameters: 
            nlp_obj (NLPPipe):
                Object of NLPPipe class. See NLPPipe.py for details
            lower (int):
                Index of most frequent word to return 
            upper (int):
                Index of least frequent word to return
        Returns:
            top_features (list):
                list of words ordered by frequency in given range
    """

    # get tokens sorted by frequency and their indices
    indices = np.argsort(nlp_obj.vectorizer.idf_)
    features = nlp_obj.vectorizer.get_feature_names()
    top_n = upper
    top_features = [features[i] for i in indices[lower:top_n]]
    return top_features

def add_stopwords(stop_words, words_to_add):
    """
    Function that adds words to a list or set of stopwords for NLP processing
        Parameters:
            current_set (list or set): 
                Self-explanatory
            words_to_add (list or set):
                Self-explanatory
        Returns:
            set_of_stopwords (set):
                New set of stopwords
    """

    # convert to list if given a set to make mutable
    stop_words = list(stop_words)
    words_to_add = list(words_to_add)

    for word in words_to_add:
        stop_words.append(word)

    set_of_stopwords = set(stop_words)
    return set_of_stopwords

def bot_policy(homepage):
    """
    Function that prints bot policy given a website homepage url
        Parameters:
            homepage (string):
                link to homepage of website of interest
        Returns:
            None:
                prints bot policy line-by-line
    """

    url = homepage + '/robots.txt'
    robots_txt = open_url(url)
    for line in robots_txt:
        print(line.decode('utf-8'))
