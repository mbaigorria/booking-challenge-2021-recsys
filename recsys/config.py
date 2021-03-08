import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SPLITS = 10
EPOCHS = 50
BATCH_SIZE = 256
EMBEDDING_SIZES = {
    'affiliate_id': (3611, 25),
    'booker_country': (5, 25),
    'checkin_day': (31, 5),
    'checkin_month': (12, 5),
    'checkin_year': (3, 5),
    'city_id': (39901, 128),
    'days_stay': (30, 5),
    'device_class': (3, 5),
    'hotel_country': (195, 25),
    'transition_days': (32, 5)
}
FEATURES_TO_ENCODE = ['city_id', 'device_class', 'affiliate_id',
                      'booker_country', 'hotel_country', 'checkin_year',
                      'days_stay', 'checkin_day', 'checkin_month',
                      'transition_days']
FEATURES_EMBEDDING = FEATURES_TO_ENCODE + ['next_' + column for column in
                                           ['affiliate_id', 'booker_country', 'days_stay', 'checkin_day']]
