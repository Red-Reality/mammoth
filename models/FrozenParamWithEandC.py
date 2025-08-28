import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser

class FrozenParamWithEandC(ContinualModel):
    """Continual learning via freezing parameters with E and C."""
    NAME = 'frozen_param_with_e_and_c'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        

    