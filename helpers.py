import numpy as np

def top_features(nlp_obj, lower, upper):
    indices = np.argsort(nlp_obj.vectorizer.idf_)
    features = nlp_obj.vectorizer.get_feature_names()
    top_n = upper
    top_features = [features[i] for i in indices[lower:top_n]]
    return top_features

def add_stopwords(current_set, words_to_add):
    stop_words = list(current_set)

    for word in words_to_add:
        stop_words.append(word)
    return set(stop_words)
