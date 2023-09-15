import math
import json
import os.path as osp

from torch.optim import lr_scheduler, AdamW
from torch import arange
from PIL import Image

def read_config_file(filename: str):
    with open(filename) as f:
        conf=json.load(f)
    
    return conf

def neat_print_dict(config, indent=0, addition=4):
    '''
    neat print dictionary
    '''
    if not isinstance(config, dict):
        raise TypeError("Argument config must be dict, got {}".format(type(config)))
    for k,v in config.items():
        if isinstance(v, dict):
            print("{}'{}': {{".format(" "*indent, k))
            neat_print_dict(v, indent=indent+addition, addition=addition)
            print("{}}}".format(" "*indent))
            continue

        print("{}'{}': {}".format(" "*indent, k, v))

def step_decay_epoch(optim,
               alpha: float,
               decay_rate: float,
               step_per_epoch: int,
               warmup_steps: int=0,
               last_epoch: int=-1,
               **kwargs):
    assert 0<alpha<1.0, \
            "Argument alpha must be in (0,1), got {}".format(alpha)
    assert warmup_steps>=0, "Warm-up steps must be geq 0, got {}".format(warmup_steps)
    
    def lr_lambda(cur_step):
        if cur_step < warmup_steps:
        # warm_up, ratio=alp+(1-alp)*(cur/warmup)
            return alpha+(1-alpha)*(float(cur_step)/float(max(1, warmup_steps)))
        
        return max(alpha, decay_rate**((cur_step-warmup_steps)//step_per_epoch))
    
    return lr_scheduler.LambdaLR(optim, lr_lambda, last_epoch)


def cosine_decay_warmup(optim,
                        alpha: float,
                        warmup_steps:float, 
                        total_steps:int,
                        cycle_by_epoch:bool=False, 
                        cycles:float=0.5,
                        last_epoch:int=-1,
                        **kwargs):
    assert 0<alpha<1.0, \
            "Argument alpha must be in (0,1), got {}".format(alpha)
    assert warmup_steps>=0, \
            "Warm-up steps must be geq 0, got {}".format(warmup_steps)
    
    def lr_lambda(cur_step):
        if cycle_by_epoch:
            cur_step=cur_step%total_steps
        if cur_step < warmup_steps:
        # warm_up, ratio=alp+(1-alp)*(cur/warmup)
            return alpha+(1-alpha)*(float(cur_step)/float(max(1, warmup_steps)))
        # cosine decay
        progress = float(cur_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return (1 - alpha) * 0.5 * (1.0 + math.cos(math.pi * float(cycles) * 2.0 * progress)) + alpha

    return lr_scheduler.LambdaLR(optimizer=optim,
                                 lr_lambda=lr_lambda,
                                 last_epoch=last_epoch)

def convert_inference_src(src: list,
                          preprocessor, 
                          style_dict, 
                          prefix_len: int):
    # src: [{"images": path, "personality": style_word, "reference": (opt)}]
    # dest:{"pixel_values":[], "prefix_ids": [], "input_ids": [好像可以不给是不是, 用默认bos试一下吧]}
    keys=src[0].keys()
    if isinstance(style_dict, str):
        # maybe a path
        assert osp.exists(style_dict), \
            "A path is given, but {} does not exist".format(style_dict)
        
        with open(style_dict) as f:
            style_dict=json.load(f)["items"]
        
    elif "items" in style_dict.keys():
        # unwrap
        style_dict=style_dict["items"]

    n_cls=len(style_dict)

    src_convert={k:list() for k in keys}
    for entity in src:
        for k in entity.keys():
            src_convert[k].append(entity.get(k, None))
    
    # src_convert: {"image": [], "personality": [], "reference": []}
    imgs=[Image.open(img) for img in src_convert["images"]]
    desc=preprocessor(images=imgs, return_tensors="pt")

    style_ids=list(map(lambda x:style_dict.get(x, n_cls-1), src_convert["personality"]))
    
    pfx_mapping=arange(0, n_cls*prefix_len).reshape((n_cls, prefix_len))
    prefix_ids=pfx_mapping[style_ids,:]
    # add prefix ids to desc
    desc["prefix_ids"]=prefix_ids

    return desc, src_convert.pop("reference")
