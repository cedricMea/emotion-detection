import sys

from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from src import config, engine, utils
from src.processing import emb_processing
from flask_cors import CORS
import tensorflow.keras as keras



app = Flask(__name__)
api = Api(app)

# to allow connections from others dns 
# CORS(app, origins="http://localhost:3000", allow_headers=[
#     "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
#     supports_credentials=True)

# CORS(app, origins="http://front:3000", allow_headers=[
#      "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
#      supports_credentials=True)

CORS(app)

# Cette ligne permet d'importer depuis un repertoire A 
# un modele scikit learn pickler dans un repertoire B
# Pickle garde dans son modèle "processing" qui n'est plus present dans le sys car remplacé
# par src.ml.processing. Il faut donc rajouter "processing" temporairement dans le sys.module
#sys.modules['processing'] = processing

# Load models when launching the web server
large_data_model = keras.models.load_model(config.LARGE_DATA_MODEL_PATH, custom_objects={"mcc_metric":utils.mcc_metric})
large_data_tokenizer = pickle.load(open(config.LARGE_DATA_TOKENIZER_PATH, "rb"))
large_label_dict = pickle.load(open(config.LARGE_DATA_LABEL_DICT_PATH, "rb"))


custom_data_model = keras.models.load_model(config.CUSTOM_DATA_MODEL_PATH, custom_objects={"mcc_metric":utils.mcc_metric})
custom_data_tokenizer = pickle.load(open(config.CUSTOM_DATA_TOKENIZER_PATH, "rb"))
custom_data_dict = pickle.load(open(config.CUSTOM_DATA_LABEL_DICT_PATH, "rb"))
#del sys.modules['processing']



class HelloWorld(Resource):
    def get(self):
        return {'/largedatamodel': 'Train with tho model train on big dataframe'}


class LargeDataModel(Resource):
    def get(self):
        sentence = request.args.get('sentence')

        # processing
        clean_sentence = utils.clean_tweets(sentence)
        clean_sentence = emb_processing.clean_text(pd.DataFrame({"text": [clean_sentence]}))
        tokenize_sentence, _ = emb_processing.tokenize_text(clean_sentence, tokenizer=large_data_tokenizer)

        preds = large_data_model.predict(tokenize_sentence).flatten()

        dict_predictions = {large_label_dict.get(key): str(preds[key]) for key in range(preds.shape[0])}
        dict_predictions["sentence"] = sentence

        return dict_predictions



class CustomDataModel(Resource):
    def get(self):
        sentence = request.args.get('sentence')

        # processing
        clean_sentence = utils.clean_tweets(sentence)
        clean_sentence = emb_processing.clean_text(pd.DataFrame({"text": [clean_sentence]}))
        tokenize_sentence, _ = emb_processing.tokenize_text(clean_sentence, tokenizer=custom_data_tokenizer)

        preds = custom_data_model.predict(tokenize_sentence).flatten()

        dict_predictions = {custom_data_dict.get(key): str(preds[key]) for key in range(preds.shape[0])}
        dict_predictions["sentence"] = sentence

        return dict_predictions




api.add_resource(HelloWorld, '/')
api.add_resource(LargeDataModel, '/largedatamodel')
api.add_resource(CustomDataModel, '/customdatamodel')




if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

# run with gunicorn : gunicorn -p localhost:5000 wsgi:app
# port par defaut pour gunicorn 8000
