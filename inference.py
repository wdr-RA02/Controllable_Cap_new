import os.path as osp
import json, torch, argparse

from model import sel_device
from utils import neat_print_dict
from utils.logger import get_now_str
from finetune_pfx_dec import load_model
from evaluate_model import inference

def img_hash_to_addr(x, img_addr, img_attr):
    x["images"]=osp.join(img_addr, x.pop("image_hash")+img_attr)

    return x

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--inference_src", 
                        type=str, default="./misc/inference_src.json")
    parser.add_argument("--load_ckpt", type=str)
    args=parser.parse_args()

    # load conf and inference_src
    with open(args.conf) as f:
        config=json.load(f)

    with open(args.inference_src) as f:
        inference_list=json.load(f)

    work_dir=config["work_dir"]
    pt_path=osp.join(work_dir, "outputs")
    device=sel_device()

    img_addr=config["dataset"]["img_path"]
    img_attr=config["dataset"]["img_attr"]
    inference_list=[img_hash_to_addr(x, img_addr, img_attr) for x in inference_list]

    # load ckpt
    model, processor, load_info=load_model(config, pt_path, device)
    print(model.get_pfx_status())

    load_ckpt=args.load_ckpt
    if load_ckpt is not None:
        print("Loading checkpoint from {}".format(load_ckpt))
        state_dict=torch.load(load_ckpt)
        missing_keys, unexp_keys=model.load_state_dict(state_dict, strict=False)
    
        print("Missing keys: {}".format(missing_keys))
        print("Unexpected keys: {}".format(unexp_keys))
    
    inf_r=inference(model, config, processor, device, inference_list)
    print("\n--------Inference results------------")
    for item in inf_r:
        neat_print_dict(item)
        print()

    # dump result
    out_name=osp.join(work_dir, "inference_{}.json".format(get_now_str()))
    with open(out_name, "w+") as f:
        json.dump(inf_r, f, indent=4)
    
    print("\n---------Inference results end-----------")
    print("Result dumped to {}".format(out_name))


