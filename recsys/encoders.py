import logging
from typing import List, Dict

import numpy as np
import pandas as pd


class LabelEncoder:
    """
    LabelEncoder similar to `sklearn.preprocessing.LabelEncoder`
    with the exception it ignores `NaN` values.

    .. todo:: Enhance this encoder with the option to set a `min_frequency`.
    """

    def fit_transform(self, col: pd.Series) -> pd.Series:
        self.rev_classes_ = dict(enumerate(sorted(col.dropna().unique())))
        self.classes_ = {v: k for k, v in self.rev_classes_.items()}
        return col.apply(lambda k: self.classes_.get(k, np.nan))

    def inverse_transform(self, col: pd.Series) -> pd.Series:
        return col.apply(lambda k: self.rev_classes_.get(k, np.nan))


class DatasetEncoder:
    """
    DatasetEncoder looks to encapsulate multiple LabelEncoder objects
    to fully transform a dataset.
    """

    def __init__(self, features_embedding: List[str]):
        self.label_encoders = {c: LabelEncoder() for c in features_embedding}

    def fit_transform(self, df: pd.DataFrame) -> None:
        """
        Transform columns in all columns given by feature_embedding.
         df:
        :return:
        """
        logging.info("Running LabelEncoder on columns")
        for column, encoder in self.label_encoders.items():
            # reserve zero index for OOV elements
            df[column] = encoder.fit_transform(df[column]) + 1
            logging.info(f"{column}: {len(encoder.classes_)}")


def get_embedding_complexity_proxy(dataset_encoder: DatasetEncoder) -> Dict:
    """
    Get embedding complexity proxy
    The idea is to find out how many bits (dimension) we need to naively encode each element in the encoder.
    It's a proxy since we have no idea which is the dimension of the underlying manifold for every feature.
    """
    return {k: (len(v.classes_), np.ceil(np.log2(len(v.classes_))))
            for k, v in dataset_encoder.label_encoders.items()}
