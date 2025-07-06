import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class GPTConfig:
    block_size: int = 512
    