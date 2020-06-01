from sklearn.compose import ColumnTransformer
from src import config, utils
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import spacy, re, json
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pandas as pd, numpy as np
from nltk.corpus import stopwords

import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

#nlp = spacy.load("en_core_web_sm")
stemmer= SnowballStemmer("english")
lemmatizer=WordNetLemmatizer()

#rint(contraction_dict)
stop_words = set(stopwords.words('english')).union(set("every"))

do_not_lemantize_words = ["cookies"]

def clean_text(data_column):
    """This function will clean the tata text with a series of retreatments

    :param data_column: The original text column
    :type data_column: pd.Series or pd.Dataframe with one column
    :return: the retreted text in a dataframe column
    :rtype: pd.Series
    """

    with open(config.CONTRACTIONS_DICT_PATH) as json_file: 
        contraction_dict = json.load(json_file)


    dataframe = pd.DataFrame(data_column).copy()
    def line_preproessing(sentence_):

        sentence = str(sentence_).lower()
        
        sentence = utils.remove_contractions(sentence,contraction_dict) 
        sentence = word_tokenize(sentence)
        #sentence = [w for w in sentence if  not w in stop_words]
        #sentence = [stemmer.stem(w) for w in sentence]
        sentence = pos_tag(sentence)
        sentence =  [utils.lemantize_word(token, lemmatizer) if token[0] not in do_not_lemantize_words else token[0] for token in sentence ]
        new_sentence = " ".join([token for token in sentence]) 
        return new_sentence 

    dataframe["text"] =  dataframe["text"].apply(lambda x: line_preproessing(x))

    return(dataframe["text"])


def tokenize_text(text_col, istraining=False, tokenizer=None):
    """Tokenize text with a keras tokenizer

    :param text_col: The cleaned text to be tokenize
    :type text_col: pd.Series or pd.Dataframe with one column
    :param istraining: weather the tokenization is for training or eval, default to False
    :type istraining: bool, Optional
    :param tokenizer: None for training and an already fit tokenizer for eval, defaults to None
    :type tokenizer: [keras.Tokenizer], optional
    :return: tuple with a numpy array containing the text input tokenized and padded
    :rtype: tuple(ndarray, keras.Tokenizer)
    """
    texts = text_col.values
    if istraining:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=config.MAX_LEN, padding="post")
    
    return (data, tokenizer)


def get_embedding_matrix(embedding_dict, tokenizer, reccurent_words=None):    
        """Build the embedding matrix
        :param embedding_dict: the word _to_vec map get from embedding file
        :type embedding_dict: dict
        :param tokenizer: A fited keras.Tokenizer
        :type tokenizer: keras.Tokenizer
        :return: embedding matrix for every tokenized words
        :rtype: numpy.ndarray
        """
        word_index = tokenizer.word_index
        
        embedding_dim = list(embedding_dict.values())[0].shape[0]
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

        #founded_word_dict = {}
        #not_founded_dict = {}
        for word, i in  word_index.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                #founded_word_dict[word] = embedding_dict.get(word)
            else: # words not found in embedding index will search with levanstein distance
                if reccurent_words is not None:
                    if word in reccurent_words:
                        found_word = utils.find_simalar_word(word, list(embedding_dict.keys()))
                        #print(f"{word} ==> {found_word}")
                        if found_word != word: #  correspondance have been found in the dict
                            embedding_matrix[i] = embedding_dict.get(found_word)

        return embedding_matrix


def read_embedded_vecs(file):
    ''' 
    Read embedded vec file and return three dict 
    Parameters :
        :file (string) : the path of the file to read
    Returns :
        :words_to_index : A dictionary which maps words  to their index in the dictionnary
        :index_to_words : The opposite
        :word_to_vec map
    '''
    with open(file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map







            