from tqdm import tqdm
from math import ceil
from typing import Iterable
from datetime import datetime as dt
from functools import wraps
import os
from torch.utils.tensorboard import SummaryWriter

# define a logger which can inject other operations in 5 steps
def get_now_str(format:str=None):
    if format is None:
        format="%Y%m%d-%H:%M:%S"
    return dt.now().strftime(format)

class TrainScheduler(object):
    def __init__(self, 
                  trainer_cfg: dict, 
                  dist_cfg:dict, 
                  log_step:int=None):
        '''
        logging and step based scheduler
        
        args:
        - trainer_cfg: self.config of trainer
        - dist_cfg: self.dist_config of trainer
        - log_step: logging step, will read from config.train.logging.log_step if not given
        '''
        # 设计成with形式
        self.train_cfg=trainer_cfg["train"]
        self.logging_conf=self.train_cfg["logging"]
        self.dist_cfg=dist_cfg
        self.dist_training=dist_cfg["distributed"]
        '''
        "logging":{
            //"eval_interval": 1,
            //"eval_every": "epoch",
            //"save_interval": 1,
            //"save_every": "epoch",
            "log_tensorboard": true,
            "log_step": 50
        }
        
        '''
        self.log_step=log_step if log_step is not None else int(self.logging_conf["log_step"])
        # use work_dir as default tboard work_dir
        self.use_tboard=self.logging_conf.get("log_tensorboard",False)
        self.tboard_dir=self.logging_conf.get("log_tboard_dir", trainer_cfg["work_dir"])

        assert isinstance(self.log_step, int), "Argument log_step requires to be int"
        self.__is_main=(self.dist_cfg.get("rank",0)==0) and (self.log_step>0)
        self.eval_interval=self.logging_conf.get("eval_interval", 0)

        self.log_items={}
    
    def __enter__(self):
        return self
        
    def _tqdm_write(self, log_str:str, force_output=False):
        if self.__is_main or force_output:
            tqdm.write(log_str)

    def before_train_begin(self, epoches:int, total_steps:int):
        self.epoches=epoches
        self.step_per_epoch=ceil(total_steps/epoches)
        self.__cur_step=0
        self._tqdm_write("\n------Training start-------")
        if not self.__is_main:
            self.writer=None
            self.pbar=None

            return        
        # 1. before train
        if self.use_tboard:
            writer_comment="train_process_{}".format(get_now_str("%y%m%d-%H%M%S"))
            self.tboard_dir=os.path.join(self.tboard_dir, writer_comment)
            os.makedirs(self.tboard_dir, exist_ok=True)
            self.writer=SummaryWriter(self.tboard_dir)
        else:
            self.writer=None
        self.pbar=tqdm(total=total_steps, position=0)
        

    def before_epoch_begin(self, cur_epoch:int, **kwargs):
        if self.__is_main:
            self.pbar.desc="epoch {}/{}".format(cur_epoch, self.epoches)
        self._tqdm_write("--------Start training epoch {}--------".format(cur_epoch))

    def time_to_eval(self, epoch:int):
        '''eval when is main machine and epoch reaches the inteval'''
        if self.eval_interval==0:
            return False
        
        return self.__is_main and epoch%self.eval_interval==0
    
    def after_iter_end(self, epoch:int, iter:int, loss, current_lr):
        '''
        loss is loss as is, no need to pass loss.item()
        '''
        # after_iter_end
        self.__cur_step+=1
        if not self.__is_main:
            return
        
        # log
        self.pbar.update(1)
        if iter%self.log_step==0:
            '''logging'''
            self.__log_epoches(iter, epoch, self.epoches, current_lr, loss)
        if self.use_tboard:
            max_lr=self.train_cfg["scheduler"]["lr"]
            self.writer.add_scalar("train/current_lr_ratio",current_lr/max_lr, self.__cur_step)
            self.writer.add_scalar("train/loss_ep{}".format(epoch), loss, iter)
        # impl per_epoch hook
        
    def update_logs(self, **kwargs):
        self.log_items.update(kwargs)

    def __log_epoches(self, cur_step, epoch, epoches, cur_lr, loss):
        '''
        input loss.item() as loss here! 
        '''
        log_format="Time: {0}, Epoch: {1}/{2}, Iter: {3:>}/{4}, lr: {5:.3e}, Loss={6:.4f}"
        additional_logs=[]
        for item in self.log_items:
            additional_logs.append("{}: {}".format(item, self.log_items[item]))

        # get current epoch and iter
        cur_step=int(cur_step)

        # display status
        cur_time=get_now_str()
        self._tqdm_write(log_format.format(cur_time, epoch, epoches, cur_step, 
                                           self.step_per_epoch, cur_lr, loss))
        if len(additional_logs)>0:
            self._tqdm_write(",".join(additional_logs))


    def after_epoch_end(self, cur_epoch:int, eval_fn=None, eval_kwargs=None, **kwargs):
        if not self.__is_main:
            return
        
        self._tqdm_write("--------End training epoch {}--------".format(cur_epoch))
        

    def after_train_end(self, **kwargs):
        self._tqdm_write("------Training done-------")
        self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.__is_main:
            self.pbar.close()
    
    
    def __exit__(self, type, value, trace):
        self.close()

        return False