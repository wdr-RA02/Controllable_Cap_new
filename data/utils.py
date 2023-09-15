import re
import os.path as osp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image

# copied from BLIP/data/utils.py
def pre_captions(caption,max_words=50):
    assert isinstance(caption, list), \
        "For processing single sentence, please use pre_caption_single() fn"
    
    processed_captions=[]

    for x in caption:
        if isinstance(x,list):
            processed_captions.append([pre_caption_single(single, max_words) for single in x])
        else:
            processed_captions.append(pre_caption_single(x, max_words))

    return processed_captions

def pre_caption_single(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~?])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption

def collate_test_set(src):
    '''
    merges additional_comments into comment
    '''

    assert "additional_comments" in src.keys()
    other_keys=[k for k in src.keys() if k not in ["comment","additional_comments"]]
    single_instance = isinstance(src["comment"],str)
    # other elements
    tgt={k:src[k] for k in other_keys}
    if single_instance:
        tgt["comment"]=[src["comment"], *src["additional_comments"]]
    else:
        tgt["comment"]=[items for items in zip(src["comment"], *src["additional_comments"])]

    return tgt

def img_hash_to_addr(src_dataset, img_addr, img_name_fmt):
    '''
    src_dataset={
        "image_hash": [paths],
        "comment":[],
        "personality":[]
    }
    '''
    if "images" in src_dataset.keys():
        return
    # 坑1: src_dataset is a dict, not dataset
    # 坑2: 使用dataloader加载的时候index是单个, 而不是一般想象中的batch

    src_dataset["images"]=[osp.join(img_addr, 
                            img_name_fmt.format(x))
                          for x in src_dataset.pop("image_hash")]
    
    return src_dataset

def image_transform(image_paths, tgt_imgsize=384):
    # perform random crop
    img_trans=transforms.Compose([
        lambda x: Image.open(x).convert("RGB"),
        transforms.RandomResizedCrop(size=tgt_imgsize, 
                                     scale=(0.8, 1), 
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    out_images=[img_trans(im) for im in image_paths]
    return out_images

def build_dataloader(src_dataset, batch, num_workers, dist_training:bool, sampler=None, shuffle=True):
    if dist_training:
        sampler=DistributedSampler(src_dataset)

    dataloader=DataLoader(src_dataset,
                    batch_size=batch, 
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=(not dist_training) and shuffle,
                    sampler=sampler)

    return dataloader