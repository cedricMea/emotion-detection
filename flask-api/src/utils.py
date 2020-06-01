import pandas as pd
import unicodedata
import re
import numpy as np
import io
import nltk
from  multiprocessing import Pool
from datetime import datetime
import string 
import tensorflow as tf

def load_data(path):

    data = pd.read_csv(path, sep=";", header=None)
    data.columns = ["text", "sentiment"]

    return data


def data_exploration(data_origin):

    data = data_origin.copy()
    print("Different polarity and number of instances")
    print(f"{data.sentiment.value_counts()}")
    data["Nb mots"] = data["text"].apply(lambda x : len(x.split(" ")))

    print("Texts lengths")
    print(data["Nb mots"].describe())


def remove_contractions(comment, contractions_dict):

    comment = str(comment).lower()
    comment = re.sub(r"'", " ",comment)
    comment = re.sub(r"(\w+) t ", "\\1t ",comment) # replace won t by wont, can t by cant...
    comment = re.sub(r" s ", " is ", comment)
    comment = re.sub(r" re ", " are ", comment)
    comment = re.sub(r"(\s)s(\s)", " is ", comment)
    comment = re.sub(r" d ", " would ", comment)
    comment = re.sub(r" ll ", " will ", comment)
    comment = re.sub(r" ve ", " have ", comment)



    for key, value in contractions_dict.items():
        comment = re.sub(r"\b{}\b".format(key), value, comment)

    comment = re.sub(r"(\s)+"," ", comment)
    return comment


def lemantize_word(post_tag_tuple, lemantizer_object):
    word = post_tag_tuple[0]
    tag = post_tag_tuple[1]
    
    pos_char = tag[0].lower() if tag[0].lower() in ['a', 'r', 'n', 'v'] else None

    if pos_char is None:
        return lemantizer_object.lemmatize(word)
    else:
        return lemantizer_object.lemmatize(word, pos=pos_char) 


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    _, _ = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def find_simalar_word(s1, list_word):
    """find similar word to s1 in list_word

    :param s1: OOV word
    :type s1: string
    :param list_word: list of words
    :type list_word: list of string
    :return: a tuple
    :rtype: [type]
    """

    # initialisation of return values
    return_word, num_changes = s1, 0
    
    unique_s1 = re.sub(r"(.)\1+", r"\1", s1) # remove repetitive char in the current word to test

    max_changes = min(3, len(unique_s1)) # the maximum of possible changes 
    
    # just a problem of a letter which is in double
    if unique_s1 in list_word:
        return unique_s1
    
    # check only in words with same beginings
    tab_word = np.array([s for s in list_word if str(s).startswith(s1[0])])
    
    # Compute Levansthein distance between unique_s1 and every word in list word
    map_obj = map(lambda x: nltk.edit_distance(unique_s1, re.sub(r"(.)\1+", r"\1", x)), tab_word)
    
    # Loop on Levansthein values and return the minimum
    for i, element in enumerate(map_obj):
        if element == 1:
            num_changes = 1 
            return_word = tab_word[i]
            break
        else:
            if element < max_changes:
                max_changes = element
                return_word = tab_word[i]
            
    return return_word              



def load_tweet_data(path="../inputs/text_emotion.csv"):
    
    data = pd.read_csv(path)
    data = data[["sentiment", "content"]]

    #data = data[data.sentiment.isin(["sadness","happiness","love","hate","anger", "surprise"])]

    printable_list = set(string.printable)
    data["text"] = data["content"].apply(lambda x: ''.join([char for char in x if char in printable_list]))

    data["text"] = data["text"].apply(lambda x : clean_tweets(x))
    data = data[["text", "sentiment"]]
    
    return data

def clean_tweets(line):
    line = str(line).lower()
    line = re.sub(r"(@|#|http|www)\S*", "", line) # delete everything that starts with @, # or http
    line = re.sub(r"'", " ", line)
    line = re.sub(r"(!|,|\?|\.|:|;|/|\.)", " ", line) # remove punctuation
    line = re.sub(r"\b\d+\w*", " ", line) # delete word starting with a number
    line = re.sub(r"(.)\1{2,}", r"\1", line) # replate repeating character by one
    
    # Replace some very common words
    line = re.sub("w/","with", line) # replace w/ by with
    line = re.sub(r".{0,1}haha[ah]*", "haha", line) #replace all ahhaha like by haha
    line = re.sub(r"bestie(\S)*", "best friend", line) # replace bestiessss by bestie
    line = re.sub("awsome", "awesome", line) # replace awsome by awesome
    
    
    line = re.sub(r"\s+", r" ", line) # replace many spaces by one
    
    # Idee d'amelioration : quand on fait le checks dans le dict si aucun n'est trouve regarder le mot le plus ressemblant
    return line.strip()



def mcc_metric(y_true, y_pred):
    maximums_indices = tf.math.argmax(y_pred, axis=1)
    predicted = tf.one_hot(maximums_indices, depth=y_pred.shape[1]) 
    true_pos = tf.math.count_nonzero(predicted * y_true)
    true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
    x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) 
        * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
    return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)
