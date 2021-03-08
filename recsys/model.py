import functools
import hashlib
import inspect
import json
import logging
from typing import List, Dict, Tuple, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from recsys import config
from recsys.config import DEVICE
from recsys.types import ModelType, WeightType, OptimizerType, BatchType, RecurrentType, FeatureProjectionType


class BookingNet(nn.Module):
    """
    BookingNet Sequence Aware Recommender System Network
    """

    def __init__(self,
                 features_embedding: List[str],
                 hidden_size: int,
                 output_size: int,
                 embedding_sizes: Dict[str, Tuple[int, int]],
                 n_layers: int = 2,
                 dropout: float = 0.3,
                 rnn_dropout: float = 0.1,
                 tie_embedding_and_projection: bool = True,
                 model_type: ModelType = ModelType.MANY_TO_MANY,
                 recurrent_type: RecurrentType = RecurrentType.GRU,
                 weight_type: WeightType = WeightType.UNWEIGHTED,
                 feature_projection_type: FeatureProjectionType = FeatureProjectionType.CONCATENATION,
                 **kwargs: List):
        """
        Args:
             features_embedding: Features to embed at each time step.
             hidden_size: Hidden size of the recurrent encoder (`LSTM` or `GRU`).
             output_size: Quantity of cities to predict.
             embedding_sizes: Sizes of each feature embedding.
             n_layers: Number of recurrent layers.
             dropout: Dropout used in our input layer.
             rnn_dropout: Dropout used in recurrent layer.
             recurrent_type: Select between `RecurrentType.GRU` or `RecurrentType.LSTM`
             tie_embedding_and_projection: If `true`, parameterize last linear layer with embedding matrix.
             feature_projection_type: Select between `FeatureCombinationType.CONCATENATION`
                or `FeatureCombinationType.MULTIPLICATION`
             model_type: The model can either only predict the last city (`ModelType.MANY_TO_ONE`) or
                predict every city in the sequence (`ModelType.MANY_TO_MANY`)
             weight_type:
                1. `WeightType.UNWEIGHTED`: Unweighted cross entropy.
                2. `WeightType.UNIFORM`: Uniform cross entropy.
                3. `WeightType.CUMSUM_CORRECTED`: Cross entropy corrected to reflect original
                    one to many weighting.
        """
        super().__init__()
        # save model arguments to re-initialize later
        model_params = inspect.getargvalues(inspect.currentframe()).locals
        if 'kwargs' in model_params:
            model_params.update(model_params['kwargs'])
            model_params.pop('kwargs')
        model_params.pop('__class__')
        model_params.pop('self')
        self.model_params = model_params

        self.features_embedding = features_embedding
        self.hidden_size = hidden_size
        self.target_variable = "next_city_id"
        self.embedding_layers = nn.ModuleDict(
            {key: nn.Embedding(num_embeddings=int(qty_embeddings) + 1,  # reserve 0 index for padding/OOV.
                               embedding_dim=int(size_embeddings),
                               max_norm=None,  # Failed experiment, enforcing spherical embeddings degraded performance.
                               norm_type=2,
                               padding_idx=0)
             for key, (qty_embeddings, size_embeddings) in embedding_sizes.items()})

        # encode every variable with the prefix `next_` to the embedding matrix of the suffix.
        self.features_dim = int(np.sum([embedding_sizes[k.replace("next_", "")][1]
                                        for k in self.features_embedding]))
        self.city_embedding_size = embedding_sizes['city_id'][1]

        self.feature_combination_type = feature_projection_type
        self.tie_embedding_and_projection = tie_embedding_and_projection
        self.recurrent_encoder = self.get_recurrent_encoder(recurrent_type, n_layers, rnn_dropout)

        if feature_projection_type == FeatureProjectionType.MULTIPLICATION:
            self.attn_weights = nn.ParameterDict(
                {key: nn.Parameter(torch.rand(1)) for key in self.features_embedding}
            )

        if self.city_embedding_size != self.hidden_size:
            logging.info(
                f"Warning: Using linear layer to reconcile output of size "
                f"{self.hidden_size} with city embedding of size {self.city_embedding_size}.")
            self.linear_to_city = nn.Linear(self.hidden_size,
                                            self.city_embedding_size,
                                            bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.city_embedding_size, output_size, bias=False)

        if self.tie_embedding_and_projection:
            # ignore first embedding, since it corresponds to padding/OOV
            self.dense.weight = nn.Parameter(self.embedding_layers['city_id'].weight[1:])

        # self.initialize_parameters()

        # other parameters
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.model_type = model_type
        self.weight_type = weight_type
        self.optimizer = None
        self.cross_entropy_weights = None

    def forward(self, batch: BatchType, seq_length: torch.Tensor):
        seq_length = seq_length.squeeze()

        # build feature map
        feature_input = self.get_feature_input(batch)
        feature_input = self.dropout(feature_input)

        # sequence encoder
        feature_input = nn.utils.rnn.pack_padded_sequence(feature_input,
                                                          seq_length,
                                                          batch_first=True,
                                                          enforce_sorted=False)
        seq_out, _ = self.recurrent_encoder(feature_input)
        seq_out, _ = nn.utils.rnn.pad_packed_sequence(seq_out,
                                                      batch_first=True)

        # reconcile encoder output size with city embedding size
        if self.city_embedding_size != self.hidden_size:
            seq_out = self.linear_to_city(seq_out)

        # create final predictions (no softmax)
        city_encoding = self.dropout(seq_out)
        dense_out = self.dense(city_encoding)
        return dense_out

    def get_feature_input(self, batch: BatchType):
        if self.feature_combination_type == FeatureProjectionType.CONCATENATION:
            return self.feature_concatenation(batch)
        else:
            return self.feature_multiplication(batch)

    def feature_concatenation(self, batch: BatchType):
        """
        Enables feature concatenation for every sequential step.
        """
        feature_list = [self.embedding_layers[k.replace("next_", "")](batch[k]) for k in self.features_embedding]
        return torch.cat(feature_list, axis=2)

    def feature_multiplication(self, batch: BatchType):
        """
        Enables feature multiplication for every sequential step.
        """
        attention_embs = [self.attn_weights[k] * self.embedding_layers[k.replace("next_", "")](batch[k])
                          for k in self.features_embedding if k != 'city_id']
        attention = functools.reduce(lambda a, b: a + b, attention_embs)
        return self.embedding_layers['city_id'](batch['city_id']) * attention

    def get_loss(self,
                 city_scores: torch.Tensor,
                 batch: BatchType,
                 seq_len: torch.Tensor,
                 device=config.DEVICE) -> torch.Tensor:
        """
        Loss function computation for the network, depending on model type:

        Args:
            1. `ModelType.MANY_TO_ONE`: Train many to one sequential model.
            2. `ModelType.MANY_TO_MANY`: Train many to many sequential model.
        """
        bs, ts = batch['city_id'].shape
        loss = self.loss(city_scores, batch['next_city_id'].view(-1) - 1)
        loss = loss.view(-1, ts)
        if self.model_type == ModelType.MANY_TO_ONE:
            return torch.sum(loss * torch.nn.functional.one_hot(seq_len - 1).to(device)) / torch.sum(seq_len)
        elif self.model_type == ModelType.MANY_TO_MANY:
            if isinstance(self.cross_entropy_weights, int):
                return torch.sum(loss) / torch.sum(seq_len)
            else:
                # TODO: Find a way to control for variance. Batches with less
                #  subsequences should have a lower weight.
                return torch.sum(self.cross_entropy_weights[:ts] * loss) / torch.sum(seq_len)
        else:
            logging.error('Invalid model type in get_loss().')

    def get_recurrent_encoder(self,
                              recurrent_type: RecurrentType,
                              n_layers: int,
                              dropout: float):
        if recurrent_type == RecurrentType.LSTM:
            return nn.LSTM(self.features_dim,
                           self.hidden_size,
                           num_layers=n_layers,
                           dropout=dropout,
                           batch_first=True)
        elif recurrent_type == RecurrentType.GRU:
            return nn.GRU(self.features_dim,
                          self.hidden_size,
                          num_layers=n_layers,
                          dropout=dropout,
                          batch_first=True)
        else:
            logging.error('Invalid recurrent encoder type in get_recurrent_encoder().')

    def set_optimizer(self, optimizer_type: OptimizerType) -> None:
        if optimizer_type == OptimizerType.ADAMW:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False)
        elif optimizer_type == OptimizerType.ADAM:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
        else:
            logging.error('Invalid optimizer type in set_optimizer().')

    def set_entropy_weights(self,
                            train_set: pd.DataFrame):
        """
        Set entropy weights for `ModelType.MANY_TO_MANY`. These weights
        depend on the `WeightType` passed in the constructor.
        """
        if self.weight_type is WeightType.UNWEIGHTED:
            self.cross_entropy_weights = 1
        elif self.weight_type in (WeightType.UNIFORM, WeightType.CUMSUM_CORRECTED):
            weights_train = dict(train_set.groupby('utrip_id').size().value_counts().items())
            weights_train = np.array([weights_train.get(k, 0) for k in range(1, 50)])
            numerator = 1 if self.weight_type == WeightType.UNIFORM else weights_train
            reweighting = numerator / np.cumsum(weights_train[::-1])[::-1]

            if np.any(np.isnan(reweighting)):
                logging.warning('Warning: NaN found in weights.')

            reweighting[np.isnan(reweighting)] = 0
            reweighting[np.isinf(reweighting)] = 0
            self.cross_entropy_weights = torch.tensor(reweighting, device=DEVICE)
        else:
            logging.error(f"Unknown weight type {self.weight_type} in set_entropy_weights()")

        logging.info(f'Weights: {self.cross_entropy_weights}')

    def initialize_parameters(self):
        """
        Network parameter initialization. Ended up using the default one.
        """
        # https://pytorch.org/docs/stable/nn.init.html
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                logging.info(f"Initializing {name}")
                nn.init.xavier_uniform_(param)

    def __str__(self):
        return json.dumps(self.model_params, indent=4, sort_keys=True)

    @property
    def hash(self):
        """
        Unique model hash for checkpoint/metrics identification.
        """
        return hashlib.md5(self.__str__().encode('utf-8')).hexdigest()[:8]


def get_model_predictions(model: BookingNet,
                          data_loader: DataLoader,
                          model_ckpt_path: str) -> Iterator[torch.FloatTensor]:
    """
    Get model predictions model checkpoint and batches data loader.
    """
    model.load_state_dict(
        torch.load(model_ckpt_path,
                   map_location=torch.device(DEVICE))
    )
    model.eval()
    with torch.no_grad():
        for batch, seq_len in data_loader:
            if DEVICE == 'cuda':
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            city_scores = model(batch, seq_len)
            city_scores = torch.bmm(
                torch.nn.functional.one_hot(seq_len - 1).unsqueeze(dim=1).type(torch.FloatTensor).to(DEVICE),
                city_scores).squeeze()
            city_scores = nn.Softmax(dim=1)(city_scores)
            yield city_scores.cpu()
