from datetime import datetime
import json
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as open_url

def get_page_list(first_url):
    """
    Function that makes a list of url strings 

    Parameters: 
    first_url (string): url string of page 1 

    Returns:
    page_list (list): list of page url strings
    """

    page_list = []
    for i in range(1, 251):
        page_list.append('{}//p{}'.format(first_url, i))
    return page_list
    
def get_all_topics(page_list):
    """
    Function that topics all thread titles for a school

    Parameters: 
    page_list (list): list of page urls strings

    Returns:
    topics (list): list of thread titles
    """
    count = 1
    topics = []
    
    for url in page_list:    
        response = requests.get(url)
        page = response.text
        soup = BeautifulSoup(page, 'html5')
        topics.append(one_page_topics(soup))

        if count % 25 == 0:
            print("Finished page {}".format(count))
        count += 1
    topics = [item for sublist in topics for item in sublist]
    return topics

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

def bot_policy(homepage):
    """
    Function that prints User-agent and disallowed bot activity

    Parameters:
    homepage (string): link to homepage of website of interest
    filters (string or list): list of keywords to print policy for

    Returns: prints bot policy line-by-line
    """

    url = homepage + '/robots.txt'
    robots_txt = open_url(url)

    initial = True
    for line in robots_txt:
        if initial == True:
            print(line.decode('utf-8'))
            initial = False
        elif "Disallow" in line.decode('utf-8'):
            print(line.decode('utf-8'))
        
def one_page_topics(soup):
    """
    Function that scrapes the thread titles displayed on one page of results
    
    Parameters: 
    soup (BeautifulSoup): soup object of the page to scrape thread topics from
    
    Returns:
    topics_json (json):  {'topic': topic string,
                             'url: url string}
    """

    topics_json = []
    for div in soup.find_all('div',  class_='Title'):
        for link in div.find_all('a'):
            topics_json.append({'topic': link.text, 
                                'url': link.get("href")})
    return topics_json
  
def page_comments(url):
    """
    Function gets comments from a page and saves them to a json

    Parameters:
    url (string): url of page to scrape

    Returns:
    lst (list): list of comment jsons

        {'date' :date,'comment': comments,
        'user_id': user_id,'user_name': user_name,
        'user_thread_count': user_thread_count,
        'user_comment_count': user_comment_count}
    """
    comments = []
    
    page = requests.get(url, timeout=100).text
    soup = BeautifulSoup(page, 'html5')
    
    # initialize all lists 
    dates = []
    comments = []
    user_id = []
    user_names = []
    user_thread_count = []
    user_comment_count = []
    lst = []
        
    # comment dates broken by time component
    for div in soup.find_all('div', class_ ="Meta Discussion DiscussionInfo"):
        for i in div.find_all('span', class_="MItem DateCreated"):
            for j in i.find_all('time'):
                split = j.get('title').split()
                dates.append({
                            'month': split[0],
                            'day': split[1],
                            'year': split[2],
                            'time': split[3]
                            })
            
    # comment text
    for div in soup.find_all('div', class_ = "Message userContent"):
        comments.append(div.text)    
    # user name
    for div in soup.find_all('div', class_ ="AuthorWrap"):
        for span in div.find_all('span', class_="Author"):
            user_names.append(span.text.replace('\n', ''))
    # user id
    for div in soup.find_all('div', class_ ="AuthorWrap"):
        for i in div.find_all('a'):
            user_id.append(i.get("data-userid"))
    # user thread count
    for div in soup.find_all('div', class_ ="AuthorWrap"):
        for span in div.find_all('span', class_ = "MItem CountDiscussions"):
            user_thread_count.append(span.text.split()[0])      
    # user comment count
    for div in soup.find_all('div', class_ ="AuthorWrap"):
        for span in div.find_all('span', class_ = "MItem CountComments"):
            user_comment_count.append(span.text.split()[0])
            
    # make json
    for i in range(0, len(comments)):
        lst.append({
            'date' :dates[i],
            'comment': comments[i],
            'user_id': user_id[i],
            'user_name': user_names[i],
            'user_thread_count': user_thread_count[i],
            'user_comment_count': user_comment_count[i]
                    })
    return lst  

def thread_comments(url):   
    """
    Function that gets all thread comments

    Parameters:
    url (string): url of first page of thread to scrape

    Returns:
    flat_list (list): list of all thread comments
    
    """ 
    all_comments = []
    page_num = 1
    
    # splitting string to insert page suffix
    base_url = url.split(".html")[0]
    test_url = base_url + "-p{}".format(page_num) + ".html"
    response = requests.get(test_url, timeout=100)
    
    while response.status_code != 404:
        test_url = base_url + "-p{}".format(page_num) + ".html"
        response = requests.get(test_url, timeout=100)
        all_comments.append(page_comments(test_url))
        page_num += 1
   
    flat_list = [item for sublist in all_comments for item in sublist]
    return flat_list

def get_forum_comms(topics_json, lower, upper):
    """"
    Function gets all comments for topics in index range lower to upper in topics_json.
    Can pass in splices for batching

    Parameters:
    topics_json (json): json with thread title and url

    Returns: saves combined json to disk and prints statement after
    """
    forum = []
    
    for topic in topics_json:
        start_time = datetime.now()
        thread = topic['topic']
        url = topic['url']
        
        try:
            forum.append({
                        'topic': thread,
                        'url': url,
                        'comments' : thread_comments(url)
                         })
        except:
            print("Thread {} failed".format(i))
            
        
        time_elapsed = datetime.now() - start_time
        print("{}: Thread {}:".format(datetime.now(), i), 'Time elapsed {}(hh:mm:ss.ms)'.format(time_elapsed))

        i += 1
    
    # for windows file structure 
    path = "C:\\Users\\Meehir\\Documents\\GitHub\\project-4-public\\"
    filename = path + "penn_data\\{}_to_{}.json".format(lower, upper - 1)
     
    with open(filename, 'w', encoding = 'utf-8') as outfile:
        json.dump(forum, outfile)
        
    print("saved {}".format(filename))

def make_scrape_list(topics_json, step):
    """
    Function outputs a list of strings that will be pasted to batch scraping

    Parameters:
    topics_json (json): json with thread names and urls
    step (int): batch size for scraping

    Returns: prints list of strings of function calls
    """
    low = 0
    high = step

    lowers = []
    uppers = []
    while low < len(topics_json):
        lowers.append(low)
        low += 100

    while high < len(topics_json):
        uppers.append(high)
        high +=  100
        
    uppers.pop(-1)
    uppers.append(len(topics_json))
    
    lims = list(zip(lowers, uppers))
    
    for i, j in lims:
        print("get_forum_comms(topics_json[{}:{}], {}, {})".format(i, j, i, j))