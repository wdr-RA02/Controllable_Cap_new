from .utils import *
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from typing import List, Dict

class ImageCaptionMetric(object):
    '''
    a custom ICMetric implemented using pycocoeval

    args:
    - sample_text: inputs中原始文本字段的key名字, 默认为comment
    - image_text: inputs中图像字段的key名字, 默认为"pixel_values"
    - mul_100: 得到的分数是否乘以100
    '''
    def __init__(self, 
                 sample_text: str="comment",
                 image_text: str="pixel_values",
                 mul_100: bool=False,
                 eos_token: str="</s>",
                 pad_token: str="<pad>"
                 ):
        print("Using pycocoeval metric currently. ")
        self.sample_text=sample_text
        self.image_text=image_text
        self.multiply=(1.,100.)[int(mul_100)]
        # reference and ground truth dicts
        self.reference=dict()
        self.ground_truth=dict()
        # init metrics
        self.eval_metrics=[
            (["BLEU_1", "BLEU_4"], Bleu(4)),
            ("ROUGE_L", Rouge()),
            ("CIDEr", Cider()),
            ("SPICE", Spice())
        ]
        self.ptb=PTBTokenizer()
        # specify tokens
        self.eos_token=eos_token
        self.pad_token=pad_token

    def add(self, outputs: List, inputs: Dict, batch_id, from_dl=False):
        '''
        add source and target to reviewer
        args:
        - outputs: inference result of the model, should be List[str]
        - inputs: source dataset, should be a dict with keys ["pixel_values"]
        - batch_id: an attribute of id for identifying each batch, should be str or int
        - from_dl: whether inputs is sourced from torch DataLoader
        
        '''
        # squeeze each ele of output["caption"]
        # image tensors, no use at all
        if self.image_text in inputs:
            del inputs[self.image_text]
        # dicts=convert_from_dataset(outputs, inputs,
        #                            pred_text=self.pred_text,
        #                            target_text=self.target_text,
        #                            sample_text=self.sample_text,
        #                            image_text=self.image_text,
        #                            eos_token=self.eos_token)
        dicts=convert_from_concap_ds(outputs, inputs[self.sample_text], 
                                     batch_id, 
                                     pad_token=self.pad_token, 
                                     eos_token=self.eos_token, from_dataloader=from_dl)
        # ref:{id: [{"caption":cap}]}
        self.reference.update(dicts[0])

        # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}
        self.ground_truth.update(dicts[1])

        # print("Input: {}".format(self.ground_truth))
        # print("Outputs: {}".format(self.reference))
        
    
    def _get_bleu_score_dict(self, 
                             label: list, 
                             score: list):
        '''
        return a flat BLEU score dict with given label

        args:
        label: BLEU item want to be included, **must be ["BLEU_k"] format**
        score: origin score list with all four BLEU scores
        return: out_dict: a dict {"BLEU_k": score[k-1]}
        '''
        out_dict=dict()
        complete_bleu=["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]
        for i,item in enumerate(complete_bleu):
            if item in label:
                out_dict.update({item: score[i]*self.multiply})
        
        return out_dict
            
    def evaluate(self):
        '''
        returns a dict of scores
        '''
        '''
        after
        ref:{id:[caption]} 
        '''
        # tokenize the content using PTBTokenizer first
        reference=self.ptb.tokenize(self.reference)
        '''
        after
        gth={id:[caption_1, caption_2, ...]}
        '''
        ground_truth=self.ptb.tokenize(self.ground_truth)
        eval_results=dict()
        for label, metric in self.eval_metrics:
            print("Evaluating {}...".format(label))
            score, _=metric.compute_score(ground_truth, reference)
            # form the dict
            if isinstance(score, list):
                result=self._get_bleu_score_dict(label, score)
            else:
                result={label:score*self.multiply}
            eval_results.update(result)
        return eval_results


    def merge(self, other: 'ImageCaptionMetric'):
        self.reference.update(other.reference)
        self.ground_truth.update(other.ground_truth)
        