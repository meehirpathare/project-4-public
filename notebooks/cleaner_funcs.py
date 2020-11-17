import re
import string

def clean(text, tokenizer, stemmer):
    
    cleaned_text = []
    for post in text:
        post = post.strip("\n")
        post = remove_emojis(post)
        post = remove_punct(post)
        post = remove_numbers(post)
        post = remove_usernames(post)
        

        cleaned_words = []
        for word in tokenizer(post):
            low_word = word.lower()
            if stemmer:
                low_word = stemmer.stem(low_word)
            cleaned_words.append(low_word)
        cleaned_text.append(' '.join(cleaned_words))
    return cleaned_text


def remove_emojis(data):
    emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoji, '', data)

def remove_punct(text):
    clean_text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return clean_text

def remove_numbers(text):
    clean_text = re.sub('\w*\d\w*', '', text)
    clean_text = re.sub('\d\w*', '', clean_text)
    return clean_text

def remove_usernames(text):
    clean_text = re.sub('@\w*', '', text)
    return clean_text

def clean_text_string(text):
    post = text.strip("\n")
    post = remove_emojis(text)
    post = remove_punct(post)
    post = remove_numbers(post)
    post = remove_usernames(post)
    post = post.lower()
    return post

def clean_list(lst):
    cleaned = []
    for post in lst:
        post = post.strip("\n")
        post = remove_emojis(post)
        post = remove_punct(post)
        post = remove_numbers(post)
        post = remove_usernames(post)
        post = post.lower()

        cleaned.append(post)
        
    return cleaned