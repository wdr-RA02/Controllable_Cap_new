import json
import torch
import os.path as osp
import random
from datasets import load_dataset
from .utils import pre_captions, img_hash_to_addr, collate_test_set, image_transform

from PIL import Image
from torch.utils.data import Dataset


class Personality_Captions(Dataset):
    def __init__(self, dataset, preprocessor, config:dict, pretrain:bool=False, **kwargs):
        super().__init__()
        # config: at least contains keys of ["style_dict", "pfx_len", "img_addr", "img_attr"]
        with open(config["style_dict"]) as f:
            self.style_dict=json.load(f)["items"]
        self.pfx_len=config["prefix_len"]
        self.n_cls=config["n_cls"]
        assert len(self.style_dict)==self.n_cls, \
            "Style number from the dict does not match that of config file. \
            Got {} and {} respectively.".format(len(self.style_dict),self.n_cls)

        # preprocessor: BlipPreprocessor
        self.preprocessor=preprocessor
        # dataset: load from datasets.load_dataset
        self.dataset=dataset
        # merge additional column into "comment"
        if "additional_comments" in self.dataset.column_names:
            self.dataset=self.dataset.map(collate_test_set, batch_size=128)

        # prefix ids w.r.o styles
        # n_cls=216, with the last being unk
        self.pfx_ids=torch.arange(0, self.n_cls*self.pfx_len)\
                    .view(-1,self.pfx_len).contiguous()
        
        self.img_addr=config["img_path"]
        self.img_name_fmt="{}%s"%(config["img_attr"])

        # others
        self.image_size=config["image_size"]
        self.split=config["split"]
        self.max_len=config.get("max_len", 30)
        # pretrain now only available for train split
        self.pretrain=pretrain and self.split=="train"

    def __getitem__(self, index):
        # 对于单int提取的情况则升维
        squeeze_list=False
        if isinstance(index, int):
            squeeze_list=True
            index=[index]

        is_train=(self.split=="train")
        item=img_hash_to_addr(self.dataset[index], self.img_addr, self.img_name_fmt)
        if is_train:
            imgs=image_transform(item["images"], self.image_size)
        else:
            imgs=[Image.open(img).convert("RGB") for img in item["images"]]

        texts=pre_captions(item["comment"], self.max_len)

        processed=self.preprocessor(images=imgs, 
                                    text=texts if self.split!="test" else None,
                                    padding="max_length",
                                    max_length=min(512-self.pfx_len, self.max_len*3),
                                    return_tensors="pt")
        
        if self.pretrain:
            neg_caps, neg_style_ids=self.__get_neg_example(item)
            processed_negs=self.preprocessor(images=None,
                                             text=neg_caps,
                                             padding="max_length",
                                             max_length=min(512-self.pfx_len, self.max_len*3),
                                             return_tensors="pt")

            processed["neg_input_ids"]=processed_negs.pop("input_ids")
            processed["neg_attention_mask"]=processed_negs.pop("attention_mask")
            processed["neg_prefix_ids"]=self.pfx_ids[neg_style_ids,:]

        # insert style pfxs
        style_ids=list(map(lambda x:self.style_dict.get(x, self.n_cls-1), item["personality"]))
        # print(style_ids)
        processed["prefix_ids"]=self.pfx_ids[style_ids,:]

        
        processed={k:v.squeeze(0) for k,v in processed.items()}
        if not is_train:
            # squeeze the sequence
            texts=texts[0] if squeeze_list else texts
            processed.update({"comment": texts})

        # "comment" in output is like [(comment_0~(batch-1)[0],...),(comment_batch[1],...),...]
        # and we desire [(comment_0[0~4]), (comment_1[0~4])]
        # so remember to zip it
        return processed

    def __get_neg_example(self, item):
        # select one negative cap from negative caps
        neg_caps=[random.choice(x) for x in item["negative_caps"]]
        neg_caps=pre_captions(neg_caps, self.max_len)
        # sample a different style for each item
        gts_styles=[self.style_dict.get(x, self.n_cls-1) for x in item["personality"]]
        neg_style_ids=[]
        for item in gts_styles:
            while True:
                neg_item=random.randint(0, self.n_cls-2)
                if neg_item != item:
                    neg_style_ids.append(neg_item)
                    break
        
        return neg_caps, neg_style_ids
        
    def __len__(self):
        return len(self.dataset)

def build_pcap_dataset(config, preprocessor, device="cpu", split="train", slice:str=""):
    '''
    load PCap dataset

    args:
    - config: ConCap.config
    - preprocessor, device
    - split: must be one of ["train", "test", "(e)val"]
    - slice: str with format of "[a:b]"
    
    '''
    print("\n-------dataset loading-------")
    split=split.lower()
    if split=="eval":
        split="val"
    if split not in ["pretrain", "train", "test", "val"]:
        raise KeyError("dataset split must be [\"train\", \"test\", \"(e)val\"]")
    
    copy_keys=["prefix_len", "n_cls", "max_len"]
    pretrain=(split=="pretrain")
    # load dataset conf
    ds_conf={
        **config["dataset"],
        **{k: config["text_model"][k] for k in copy_keys},
        "image_size": config["vision_model"].get("image_size", 384),
        "split": "train" if pretrain else split
    }
    
    ds_file=osp.join(ds_conf["dataset_path"],ds_conf["{}_json".format(ds_conf["split"])])
    dataset=load_dataset("json", data_files=ds_file, split="train{}".format(slice))
    # we need config.
    tgt_dataset=Personality_Captions(dataset, preprocessor, config=ds_conf, pretrain=pretrain)
    
    print("\nLoad {} split from {}".format(split, ds_file))
    if slice != "":
        print("Slice: {}".format(slice))
    print("Length: {}".format(len(tgt_dataset)))
    print("-------dataset load done-------")

    return tgt_dataset



