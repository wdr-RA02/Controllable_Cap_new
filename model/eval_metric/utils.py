import os
from functools import partial
from string import punctuation
from typing import Dict, Iterable, List, Union


def pop_empty(inputs: Union[str,List[str]],
        pad_token="<pad>",
        eos_token="</s>")->List[Dict[str, List]]:
    '''
    将List[str]或单条str转换为[{"caption": cap_i}]的形式
    '''
    new_list=list()
    if isinstance(inputs,str):
        inputs=[inputs]
    assert isinstance(inputs, Iterable), \
        "inputs expected list or str input, got {}".format(type(inputs))
    for one_str in inputs:
        if len(one_str.replace(" ","").replace(".",""))>0:
            # for each caption C: C->{"caption":c}
            transtab = str.maketrans({key: None for key in punctuation})
            
            new_list.append({"caption":one_str.translate(transtab)\
                             .replace(eos_token, "").replace(pad_token, "").strip()})
    return new_list


def convert_from_dataset(outputs,
                         inputs,
                         pred_text: str="caption",
                         target_text: str="labels",
                         sample_text: str="samples",
                         image_text: str="image",
                         eos_token: str="</s>"):
    '''
    将dataset的输入转换为pycocoevalcap需要的输入格式
    具体为: 
    ref={id: [{"caption":cap}]}
    gt={id: [{"caption":cap_1}, {"caption":cap_2}...]}

    args: 
    outputs: 模型的输出, 包含参考caption
    inputs: 模型的输入, 包含ground truth
    *_text: inputs中各字段的名称

    return: reference, ground_truth
    '''
    # use filename as image_id
    image_ids=[os.path.split(k[image_text])[-1] for k in inputs[sample_text]]
    # specify eos_token to truncate the end of ground_truth
    pop_empty_eos=partial(pop_empty, eos_token=eos_token)
    # ref:{id: [{"caption":cap}]}
    reference={i:caps for i,caps in zip(image_ids,map(pop_empty_eos, outputs[pred_text])) \
                    if len(caps)>0}

    # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}
    ground_truth={i:caps for i,caps in zip(image_ids,map(pop_empty_eos, inputs[target_text])) \
                    if len(caps)>0}
    
    return reference, ground_truth


def convert_from_concap_ds(ref: list, gts:list, 
                           batch_id, eos_token="[SEP]", pad_token="[PAD]", from_dataloader:bool=False):
    '''
    args:
    - ref=[1,2,...b]
    - gts=[[1_0~b_0], [1_1~b_1], ..., [1_4~b_4]] if from_dataloader

    returns: 
    - ref={id: [{"caption":cap}]}
    - gt={id: [{"caption":cap_1}, {"caption":cap_2}...]}
    '''
    if from_dataloader:
        # convert to [[1_0~4], [2_0~4], ..., [b_0~4]]
        gts=list(zip(*gts))
    
    assert len(ref)==len(gts), \
        "ref and gts got difference length {} and {}, but it is expected to be equal.".format(len(ref), len(gts))
    # specify eos_token to truncate the end of ground_truth
    pop_empty_eos=partial(pop_empty, eos_token=eos_token, pad_token=pad_token)
    index="%s_{}"%(str(batch_id))

    reference={index.format(i):cap for i, cap in zip(range(len(ref)), map(pop_empty_eos, ref))}
    ground_truth={index.format(i):cap for i, cap in zip(range(len(gts)), map(pop_empty_eos, gts))}

    return reference, ground_truth