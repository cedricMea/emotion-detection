from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from src import config, utils
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import json
import pandas as pd
from nltk.corpus import stopwords

stemmer= SnowballStemmer("english")
lemmatizer=WordNetLemmatizer()
with open(config.CONTRACTIONS_DICT_PATH) as json_file: 
    contraction_dict = json.load(json_file)

stop_words = set(stopwords.words('english')).union(set("every"))

# TFIDF - Vectorizer
def clean_text(data_column):

    dataframe = pd.DataFrame(data_column)
    def line_preproessing(sentence_):

        sentence = str(sentence_).lower()
        #print(sentence)
        sentence = utils.remove_contractions(sentence,contraction_dict) 
        #print(sentence)
        sentence = word_tokenize(sentence)
        sentence = [w for w in sentence if  not w in stop_words]
        # print(sentence)
        sentence = [stemmer.stem(w) for w in sentence]
        sentence = pos_tag(sentence)
        sentence =  [utils.lemantize_word(token, lemmatizer) for token in sentence]
        # print(sentence)
        #print(sentence)
        new_sentence = " ".join([token for token in sentence]) 
        return new_sentence 
    dataframe["text"] =  dataframe["text"].apply(lambda x: line_preproessing(x))

    #print(dataframe["text"])

    return(dataframe["text"])


def data_preprocess():

    
    clean_text_transformer = FunctionTransformer(
        func=clean_text
    )

    vectorizer = TfidfVectorizer()

    # Print(vectorizer.vocabulary_)
    # SVD vectorizer
    svd_vectorizer =TruncatedSVD(n_components=config.MAX_LEN)

    # TF-IDF + SVD = LSA 
    lsa_pipeline = Pipeline(
        steps=[
            ('preprocess_func', clean_text_transformer),
            ('tf_idf_vectorizer', vectorizer) ,
           ('svd_vectorizer', svd_vectorizer)
        ]
    )


    # Mettre en place un un pipeline pour chaque colonne
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_preprocess', lsa_pipeline, config.TEXT_COL), # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        ],
        remainder='passthrough'
    )

    return (preprocessor)