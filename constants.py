from logger import get_logger

MODEL_PHOBERT_BASE = "phobert_base"
MODEL_PHOBERT_LARGE = "phobert_large"
MODEL_BILSTM = "bilstm"
MODEL_BILSTM_CRF = "bilstm_crf"
MODEL_PHOBERT_BASE_BILSTM = "phobert_base_bilstm"
MODEL_PHOBERT_BASE_BILSTM_CRF = "phobert_base_bilstm_crf"
MODEL_PHOBERT_LARGE_BILSTM = "phobert_large_bilstm"
MODEL_PHOBERT_LARGE_BILSTM_CRF = "phobert_large_bilstm_crf"

AVAILABLE_MODELS = [
    MODEL_BILSTM,
    # MODEL_BILSTM_CRF,
    # MODEL_PHOBERT_BASE,
    # MODEL_PHOBERT_LARGE,
    # MODEL_PHOBERT_BASE_BILSTM,
    MODEL_PHOBERT_BASE_BILSTM_CRF,
    # MODEL_PHOBERT_LARGE_BILSTM,
    # MODEL_PHOBERT_LARGE_BILSTM_CRF
]


def check_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        logger = get_logger()
        logger.info((f"Model {model_name} is not supported. "
                     "Please select one of the available "
                     "models {AVAILABLE_MODELS}."))
        return False
    return True


RESOURCE_PHOBERT_BASE = "resource/models/phobert_base"
RESOURCE_PHOBERT_LARGE = "resource/models/phobert_large"
RESOURCE_BILSTM = "resource/models/bilstm"
RESOURCE_BILSTM_CRF = "resource/models/bilstm_crf"
RESOURCE_PHOBERT_BASE_BILSTM = "resource/models/phobert_base_bilstm"
RESOURCE_PHOBERT_BASE_BILSTM_CRF = "resource/models/phobert_base_bilstm_crf"
RESOURCE_PHOBERT_LARGE_BILSTM = "resource/models/phobert_large_bilstm"
RESOURCE_PHOBERT_LARGE_BILSTM_CRF = "resource/models/phobert_large_bilstm_crf"

MAX_TOKEN_LEN = 250
BILSTM_EMB_DIM = 768
BILSTM_NUM_UNITS = 128
DROP_OUT = 0.1
