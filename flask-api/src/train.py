import os, sys
import pandas as pd
import config
from src.ml import utils, engine, processing, config 
import numpy as np
import pickle
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

# get_ipython().run_line_magic('load_ext', 'autoreload') available in the notebook
# get_ipython().run_line_magic('autoreload', '2')




def run():
    # Load data 


    # Load data 
    data_train = utils.load_data(config.TRAIN_PATH)
    data_val = utils.load_data(config.VAL_PATH)

    data_initial = pd.concat([data_train, data_val])

    # Load tweets
    tweet_data = utils.load_tweet_data()


    # Merge two dataframes sources
    all_data = pd.concat([data_initial, tweet_data])

    sentiment_map = {
        "surprise": "joy",
        "happiness": "joy",
        "love": "love",
        "sadness": "sadness",
        "hate": "anger",
        "anger": "anger",
        "joy": "joy",
        "love": "love",
        "fear": "fear"  
    }

    all_data["sentiment"] = list(map(sentiment_map.get, all_data["sentiment"].values))

    data_merge = all_data.copy()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data_merge.drop(config.TARGET_COL, axis=1),
        data_merge[config.TARGET_COL],
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=data_merge[config.TARGET_COL].values
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    y_val_encoded = label_encoder.transform(y_val)

    class_weight = {0: 1.799, 1: 1 , 2: 6.568, 3: 2.477, 4: 4.830 }

    #


if __name__=="__main__":
    run()