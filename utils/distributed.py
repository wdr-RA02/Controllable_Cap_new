import os
import torch
from torch.backends import cudnn
import torch.distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import neat_print_dict

def init_seed(seed):
    import random
    import numpy as np
    
    # fix the seeds
    seed=seed+get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    cudnn.benchmark=True
    

def init_dist():
    dist_config={}
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist_config["distributed"]=True
        dist_config["rank"]=int(os.environ["RANK"])
        dist_config["world_size"]=int(os.environ["WORLD_SIZE"])
        dist_config["gpu"]=int(os.environ["LOCAL_RANK"])
    else:
        dist_config["distributed"]=False
        print("Distributed Training is disabled.")
        return dist_config

    # init groups
    torch_dist.init_process_group("nccl")
    torch.cuda.set_device(dist_config["gpu"])
    # disable print in other machines
    setup_for_distributed(get_rank()==0)
    # print config
    print("Distributed training enabled.")
    print("-----------Dist config-----------")
    neat_print_dict(dist_config)
    print("---------Dist config done----------")
    
    return dist_config


def is_dist_avail_and_initialized():
    if not torch_dist.is_available():
        return False
    if not torch_dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch_dist.get_rank()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    # disable hf logs in non_master machines
    from transformers.utils import logging
    if not is_master:
        logging.set_verbosity(logging.CRITICAL)
        logging.disable_progress_bar()
