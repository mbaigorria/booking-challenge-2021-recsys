from enum import Enum
from typing import Dict

import torch


class ModelType(str, Enum):
    MANY_TO_ONE = 1
    MANY_TO_MANY = 2


class WeightType(str, Enum):
    UNWEIGHTED = 1
    UNIFORM = 2
    CUMSUM_CORRECTED = 3


class RecurrentType(str, Enum):
    GRU = 1
    LSTM = 2


class FeatureProjectionType(str, Enum):
    CONCATENATION = 1
    MULTIPLICATION = 2


class OptimizerType(str, Enum):
    ADAM = 1
    ADAMW = 2


BatchType = Dict[str, torch.Tensor]
