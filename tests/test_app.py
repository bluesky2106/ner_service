from ner_service.app import home, extract_entities
from ner_service.constants import MODEL_PHOBERT_BASE_BILSTM_CRF
from flask import request

def test_home():
    res = home()
    assert res == "Homepage"


def test_extract_entities():
    pass