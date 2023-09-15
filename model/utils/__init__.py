import torch
from transformers import logging
from transformers import BlipForConditionalGeneration as BlipBase
from .pretrain_utils import *


def process_vit_pos_emb(src_pretrain:str, tgt_model:BlipBase):
    '''
    process the mismatch caused by changing shape

    args:
    - src_pretrain: ckpt used to load positional embedding
    - tgt_model: target model
    '''
    verbosity=logging.get_verbosity()

    # shut up tf logger
    logging.set_verbosity(logging.CRITICAL)

    pos_emb_key="vision_model.embeddings.position_embedding"
    
    print("------Processing mismatch position embedding------")
    print("Loading source model from {}...".format(src_pretrain))
    src_model=BlipBase.from_pretrained(src_pretrain)
    src_state_dict=src_model.state_dict()

    src_embed=src_state_dict[pos_emb_key].clone()
    src_shape=src_embed.shape
    hid_size=src_shape[-1]
    src_patch_num=int(src_model.vision_model.embeddings.num_patches)
    src_size=int(src_patch_num**0.5) 
    tgt_patch_num=int(tgt_model.vision_model.embeddings.num_patches)
    tgt_size=int(tgt_patch_num**0.5)

    extra_token_num=src_shape[-2]-src_patch_num
    del src_model
    if src_size!=tgt_size:
        print("Converting pos_emb src from {} -> {}...".format(src_patch_num,tgt_patch_num))
        # extract [CLS] from src_token
        src_extra_embed=src_embed[:,:extra_token_num]
        # pos_embed: [b, src_size**2, hid_size]
        pos_embed=src_embed[:, extra_token_num:]
        # convert pos_embed to [b, hid, src_size, src_size]

        pos_embed=pos_embed.reshape(-1, src_size, src_size, hid_size).permute(0,3,1,2)
        # interpolate
        pos_embed=torch.nn.functional.interpolate(pos_embed, size=(tgt_size, tgt_size), 
                                mode="bicubic", align_corners=False)
        pos_embed=pos_embed.permute(0,2,3,1).flatten(1,2)
        # pos_embed: [b, tgt_size, tgt_size, h]
        new_pos_embed=torch.cat([src_extra_embed, pos_embed], dim=1)
        print("Loading new pos emb to target_model...")
        # load the new pos_embed to tgt_model
        tgt_model.load_state_dict({pos_emb_key: new_pos_embed}, strict=False)
    else:
        print("position embedding layers have same shape, so no operation required")
    
    print("------Position embedding interpolation done-------\n")
    # restore verbosity
    logging.set_verbosity(verbosity)

    return tgt_model



def process_vit_pos_emb_dict(src_state_dict, tgt_model:BlipBase):
    '''
    process the mismatch caused by changing shape

    args:
    - src_pretrain: ckpt used to load positional embedding
    - tgt_model: target model
    '''
    verbosity=logging.get_verbosity()

    # shut up tf logger
    logging.set_verbosity(logging.CRITICAL)

    pos_emb_key="vision_model.embeddings.position_embedding"
    
    print("------Processing mismatch position embedding------")

    src_embed=src_state_dict[pos_emb_key].clone()
    src_shape=src_embed.shape
    hid_size=src_shape[-1]
    src_patch_num=int(src_state_dict[pos_emb_key].shape[1]-1)
    src_size=int(src_patch_num**0.5) 
    tgt_patch_num=int(tgt_model.vision_model.embeddings.num_patches)
    tgt_size=int(tgt_patch_num**0.5)

    extra_token_num=src_shape[-2]-src_patch_num
    if src_size!=tgt_size:
        print("Converting pos_emb src from {} -> {}...".format(src_patch_num,tgt_patch_num))
        # extract [CLS] from src_token
        src_extra_embed=src_embed[:,:extra_token_num]
        # pos_embed: [b, src_size**2, hid_size]
        pos_embed=src_embed[:, extra_token_num:]
        # convert pos_embed to [b, hid, src_size, src_size]

        pos_embed=pos_embed.reshape(-1, src_size, src_size, hid_size).permute(0,3,1,2)
        # interpolate
        pos_embed=torch.nn.functional.interpolate(pos_embed, size=(tgt_size, tgt_size), 
                                mode="bicubic", align_corners=False)
        pos_embed=pos_embed.permute(0,2,3,1).flatten(1,2)
        # pos_embed: [b, tgt_size, tgt_size, h]
        new_pos_embed=torch.cat([src_extra_embed, pos_embed], dim=1)
        print("Loading new pos emb to target_model...")
        # load the new pos_embed to tgt_model
        tgt_model.load_state_dict({pos_emb_key: new_pos_embed}, strict=False)
    else:
        print("position embedding layers have same shape, so no operation required")
    
    print("------Position embedding interpolation done-------\n")
    # restore verbosity
    logging.set_verbosity(verbosity)

    return tgt_model
