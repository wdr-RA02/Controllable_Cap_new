import os
import torch
import argparse
from tqdm import tqdm
from model import ConCapPretrainedModel
from utils import read_config_file
from model.utils import process_vit_pos_emb_dict
from model.utils.pretrain_utils import rename_key
from transformers import BlipProcessor, BlipConfig

def add_tokens(processor:BlipProcessor):
    tokenizer=processor.tokenizer
    # add [DEC] and [ENC]
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

    return processor  

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="The folder that stored other HF config file")
    parser.add_argument("--conf_file", type=str, required=True,
                        help="config.json")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="The pretrained checkpoint of BLIP")
    parser.add_argument("--output_dir", type=str,
                        help="Where to save the models")
    args=parser.parse_args()

    save_dir=args.output_dir
    if save_dir is None:
        save_dir=os.path.join(args.model_dir, "converted")
    
    if os.path.isfile(save_dir):
        raise RuntimeError("{} is a file, specify another output_dir! ".format(save_dir))
    
    # read config file
    conf=read_config_file(args.conf_file)
    # add special tokens to tokenizer
    processor=BlipProcessor.from_pretrained(args.model_dir)
    processor=add_tokens(processor)
    tokenizer=processor.tokenizer

    # load ckpt
    ckpt=torch.load(args.ckpt, map_location="cpu")["model"]
    model_cfg:BlipConfig=BlipConfig.from_pretrained(args.model_dir)
    model_cfg.text_config.enc_token_id=tokenizer.enc_token_id
    model_cfg.vision_config.image_size=conf["vision_model"]["image_size"]

    # rename
    print("Processing state dict...\n")
    with torch.no_grad():
        new_dict=ckpt.copy()
        for key in tqdm(ckpt.keys()):
            item=new_dict.pop(key)
            new_key=rename_key(key)
            new_dict[new_key]=item
    
    # load model
    model=ConCapPretrainedModel(model_cfg, 
                                pfx_config_dict=conf["text_model"])
    model.text_encoder.resize_token_embeddings(len(tokenizer))
    model.text_decoder.resize_token_embeddings(len(tokenizer))
    missing, unexpected=model.load_state_dict(new_dict, strict=False)

    print("Missing keys: {}".format(missing))
    print("Unexpected keys: {}".format(unexpected))

    # interpolate ViT
    print("Processing ViT position embedding interpolation...\n")
    model=process_vit_pos_emb_dict(new_dict, model)

    # save models
    print("Saving model to {} ...\n".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    model_cfg.save_pretrained(save_dir)
    
    # don't save uninited keys
    new_state_dict={k:v for k,v in model.state_dict().items() if k not in missing}
    model.save_pretrained(save_dir, state_dict=new_state_dict)
    processor.save_pretrained(save_dir)

    print("Save done")
