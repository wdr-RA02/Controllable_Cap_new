from transformers import BlipConfig, BlipProcessor, BlipForConditionalGeneration as BlipBase
from .concap_pretrain import ConCapPretrainedModel
from .concap_model import ConCapModel

def sel_device():
    import torch.cuda as cuda

    return ["cpu", "cuda"][cuda.is_available()]