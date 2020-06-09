
from src import config, utils
import json


class TestRemoveContractions(object):

    def test_remove_contractions_with_contraction(self):
        contraction_dict = json.load(open(config.CONTRACTIONS_DICT_PATH))
        assert utils.remove_contractions("i wasnt in the kitchen", contraction_dict) == "i was not in the kitchen"



class TestSimilarWord():

    def test_avec_mot_similaire_present(self):
        mot = "helo"
        list_mots = ["hello", "eat", "dance"]
        assert utils.find_simalar_word(mot, list_mots) == "hello"
    
    def test_sans_mot_similaire_present(self):
        mot = "helo"
        list_mots = ["eat", "dance"]
        assert utils.find_simalar_word(mot, list_mots) == "helo"


