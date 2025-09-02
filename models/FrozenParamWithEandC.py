from argparse import Namespace
import os
from backbone import MammothBackbone
from datasets.utils.continual_dataset import ContinualDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser,add_rehearsal_args
from utils.buffer import Buffer

class FrozenParamWithEandC(ContinualModel):
    """Continual learning via freezing parameters with E and C."""
    NAME = 'frozen_param_with_e_and_c'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--freeze_mask_path',type=str, required=True,
                            help='Path to the freeze mask file')
        parser.add_argument('--freeze_rate',type=float, default=0.1,
                            help='Rate of parameters to freeze')
        parser.add_argument('--replay_train_epochs', type=int, default=5,
                            help='Number of epochs to train on replay data')
        
        parser.add_argument('--alpha',dtype = float,default=0.5)
        parser.add_argument('--beta',dtype = float,default= 2)
        
        # 添加回放参数
        add_rehearsal_args(parser)
        return parser
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.freeze_masks = None
        self.last_param = None

        # 回放方法
        self.buffer = Buffer(self.args.buffer_size)
    def begin_task(self, dataset: ContinualDataset) -> None:
        if self.freeze_masks is None:
            self.freeze_masks = {
                name:torch.ones_like(param,dtype=torch.float32,device=self.device)
                for name,param in self.net.named_parameters()
            }
    def end_task(self, dataset):
        '''
        任务结束后：
        1. 用当前任务的数据更新回放的buffer
        2. 保存当前任务的参数
        3. 更新冻结掩码
        '''
        # 更新回放的buffer（参考agem）
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )
        # 保存这一次参数为下一次任务准备
        self.last_param =copy.deepcopy(self.net.state_dict())



    def save_freeze_mask(self, path):
        """保存冻结掩码"""
        if self.freeze_masks is not None:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            torch.save(self.freeze_masks, path)
        else:
            raise ValueError("Freeze masks are not set. Please run the training process first.")

    def load_freeze_mask(self, path):
        """加载冻结掩码"""
        if os.path.exists(path):
            self.freeze_masks = torch.load(path, map_location=self.device)
            
        else:
            self.freeze_masks={
                name:torch.ones_like(param,dtype=torch.float32,device=self.device)
                for name,param in self.net.named_parameters()
            }
        return self.freeze_masks
    def froze_param(self):
        """
        将变化最大的前百分之k的参数对应的freeze_mask设为0
        """
        if self.freeze_masks == None:
            raise ValueError("freeze_masks is None")
        with torch.no_grad():
            delta_param = copy.deepcopy(self.net.state_dict())
            if self.last_param is not None:
                for name,param in self.net.state_dict():
                    delta_param[name] = torch.abs(param-self.last_param[name])
            param_vector = torch.cat([param.view(-1)for param in delta_param])

            # 相对比率正则化
            param_vector = param_vector / (torch.mean(param_vector) + 1e-10)
            max_param = torch.max(param_vector)
            min_param = torch.min(param_vector)
            normalized_vector = (param_vector - min_param) / (max_param - min_param + 1e-10) * (self.args.beta - self.args.alpha) + self.args.alpha

            # 将变化最大的前百分之k的参数对应的freeze_mask设为0
            k = int(self.args.freeze_rate * param_vector.numel())
            if k < 1:
                return
            _, idx = torch.topk(normalized_vector, k)
            mask_vector = torch.ones_like(param_vector)
            mask_vector[idx] = 0
            # 将vector转为dict形式的mask
            pointer = 0
            for name,param in self.net.named_parameters():
                num_param = param.numel()
                self.freeze_masks[name] = mask_vector[pointer:pointer+num_param].view_as(param).to(self.device)
                pointer += num_param
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        '''
        训练步骤
        1. 使用当前任务的数据进行训练
        2. 冻结重要权重后，使用回放数据微调
        '''
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        self.opt.step()
        
        # 使用回放数据进行微调
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            self.net.zero_grad()
            buf_outputs = self.net(buf_inputs)
            replay_loss = self.loss(buf_outputs, buf_labels)
            replay_loss.backward()
            # 冻结
            if self.freeze_masks is None:
                raise ValueError("freeze_masks is None")
            self.froze_param()
            # log:检查还有多少可被训练的参数
            froze_vector = torch.cat([mask.view(-1)for mask in self.freeze_masks])
            num_trainable = torch.sum(froze_vector).item()
            print(f"Number of trainable parameters after freezing: {num_trainable}/{froze_vector.numel()}")
            if self.freeze_masks is not None:
                # 可能有问题
                for (param,mask) in zip(self.net.parameters(),self.freeze_masks.values()):
                    param.grad *= mask

            self.opt.step()
        return loss.item()



                