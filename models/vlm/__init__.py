from models.vlm.cogvlm import CogVLM
from models.vlm.gemini import Gemini
from models.vlm.omg_llava import OMG_LLaVA
from models.vlm.sa2va import Sa2VA
from models.vlm.glamm import GLaMM
from models.vlm.llava_ov import LLaVA_OV
from models.vlm.lisa import LISA
from models.vlm.flmm import FLMM

MODEL_REGISTRY = {
    'cogvlm': CogVLM,
    'gemini': Gemini,
    'omg_llava': OMG_LLaVA,
    'sa2va': Sa2VA,
    'glamm': GLaMM,
    'llava_ov': LLaVA_OV,
    'lisa': LISA,
    'flmm': FLMM
}

def get_model(model_name, args):
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not supported")
    else:
        return MODEL_REGISTRY[model_name](args)