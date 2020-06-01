import os

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
TRAIN_PATH = os.path.join(BASE_PATH, "inputs/train.txt")
VAL_PATH = os.path.join(BASE_PATH, "inputs/val.txt")


MODEL_PATH = os.path.join(BASE_PATH, "models")


LARGE_DATA_DIR =  os.path.join(MODEL_PATH, "large_data")
LARGE_DATA_MODEL_PATH = os.path.join(LARGE_DATA_DIR,"large_data_model")
LARGE_DATA_TOKENIZER_PATH = os.path.join(LARGE_DATA_DIR,"tokenizer.pkl")
LARGE_DATA_LABEL_DICT_PATH = os.path.join(LARGE_DATA_DIR,"large_data_label_dict.pkl")

CUSTOM_DATA_DIR = os.path.join(MODEL_PATH, "custom_data")
CUSTOM_DATA_MODEL_PATH = os.path.join(CUSTOM_DATA_DIR,"custom_data_model")
CUSTOM_DATA_TOKENIZER_PATH = os.path.join(CUSTOM_DATA_DIR,"tokenizer.pkl")
CUSTOM_DATA_LABEL_DICT_PATH = os.path.join(CUSTOM_DATA_DIR,"custom_data_label_dict.pkl")


EMBEDDING_FILE_PATH = os.path.join(BASE_PATH, "inputs/glove/glove.6B.50d.txt")
CONTRACTIONS_DICT_PATH = os.path.join(BASE_PATH, "inputs/contractions.json")



MAX_LEN = 40

TARGET_COL = "sentiment"
TEXT_COL = "text"
