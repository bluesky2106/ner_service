from ner_service.constants import MODEL_PHOBERT_BASE_BILSTM_CRF, check_model


def test_check_model():
    # happy case
    result = check_model(MODEL_PHOBERT_BASE_BILSTM_CRF)
    assert result is True

    # sad case
    result = check_model("phobert_crf")
    assert result is False
