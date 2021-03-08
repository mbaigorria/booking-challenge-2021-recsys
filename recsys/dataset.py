import numpy as np
import pandas as pd

from recsys.paths import get_resources_path

_NEXT_CITY_COLUMNS = ['city_id', 'affiliate_id',
                      'booker_country', 'days_stay',
                      'checkin_day']


def build_dataset(reserved_obs: int = 10000) -> pd.DataFrame:
    """
    Builds dataset by unifying training and test set.

    :return: pd.DataFrame with unified dataset.
    """
    train_set = pd.read_csv(get_resources_path('booking_train_set.csv'), index_col=0,
                            dtype={'user_id': 'int32', 'city_id': 'int32'},
                            parse_dates=['checkin', 'checkout']).sort_values(by=['utrip_id', 'checkin'])
    test_set = pd.read_csv(get_resources_path('booking_test_set.csv'),
                           dtype={'user_id': 'int32', 'city_id': 'int32'},
                           parse_dates=['checkin', 'checkout']).sort_values(by=['utrip_id', 'checkin'])

    # create dataset identifiers and homogenize dataframes
    train_set['train'] = 1
    test_set['train'] = 0
    test_set.drop(columns=['row_num', 'total_rows'], inplace=True)
    test_set['city_id'] = test_set['city_id'].replace({0: np.nan})

    # reserve observations for sanity check
    train_set['reserved'] = np.arange(len(train_set)) <= reserved_obs

    # unify datasets
    dataset = pd.concat([train_set, test_set])

    # create some time features
    dataset['days_stay'] = (dataset['checkout'] - dataset['checkin']).dt.days - 1
    dataset['checkin_day'] = dataset['checkin'].dt.dayofweek
    dataset['checkin_month'] = dataset['checkin'].dt.month
    dataset['checkin_year'] = dataset['checkin'].dt.year

    # create transition time feature
    dataset['prev_checkout'] = dataset.groupby('utrip_id')['checkout'].shift(periods=1)
    dataset['transition_days'] = (dataset['checkout'] - dataset['prev_checkout']).dt.days - 1
    dataset['transition_days'].fillna(0, inplace=True)
    dataset.drop(columns="prev_checkout", inplace=True)
    return dataset


def set_future_features(df: pd.DataFrame) -> None:
    """
    Add features about the next city to the dataframe.
    """
    for column in _NEXT_CITY_COLUMNS:
        df['next_' + column] = df.groupby('utrip_id')[column].shift(periods=-1)


def get_training_set_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get training set by ignoring reserved test set observations.
    """
    return df[df.reserved != True]


def get_test_set_from_dataset(df: pd.DataFrame,
                              sequence_length: int = 3) -> pd.DataFrame:
    """
    Get test set from unified dataframe and constrain the minimum
    sequence length to avoid a test/submissions set distribution mismatch.
    """
    test_set = df[df.reserved == True]
    return min_sequence_length_transformer(test_set, sequence_length)


def get_submission_set_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get submission set from dataset, filtering `NaN` cities that appeared
    when merging the training and test set.

    .. warning::
        You should create the submission set before filtering `NaN`.
    """
    submission_set = df[(df.train == 0) & (~df.city_id.isna())]
    assert len(submission_set) == 308005
    return submission_set


def min_sequence_length_transformer(df: pd.DataFrame,
                                    sequence_length: int = 3) -> pd.DataFrame:
    """
    Constrains the minimum trip length to `sequence_length`.
    """
    return df.groupby('utrip_id').filter(lambda x: len(x) >= sequence_length)
