import logging

from recsys import config
from recsys.dataset import build_dataset, get_training_set_from_dataset, \
    get_test_set_from_dataset, set_future_features, get_submission_set_from_dataset
from recsys.dataset_loader import get_dataset_and_dataloader, batches_to_device
from recsys.encoders import DatasetEncoder
from recsys.experiments import get_base_configuration, get_model_performance_data, \
    filter_results_table
from recsys.model_train import train_model_for_folds
from recsys.paths import get_path
from recsys.plot import get_plot_from_accuracy
from recsys.utils import get_distribution_by_pos, print_gpu_usage, get_final_submission

if __name__ == "__main__":
    # build and encode dataset
    dataset = build_dataset(reserved_obs=30000)
    de = DatasetEncoder(config.FEATURES_TO_ENCODE)
    de.fit_transform(dataset)
    set_future_features(dataset)

    submission_set = get_submission_set_from_dataset(dataset)

    # keep only observations before the last visit
    dataset = dataset[~dataset.next_city_id.isna()]

    # split training and test set from dataset
    train_set = get_training_set_from_dataset(dataset)
    test_set = get_test_set_from_dataset(dataset)

    logging.info(f"Training set: {train_set.shape}")
    logging.info(f"Test set: {test_set.shape}")
    logging.info(f"Dataset: {dataset.shape}")

    logging.info(get_distribution_by_pos(dataset=dataset,
                                         train_set=train_set[train_set.train == 1],
                                         test_set=test_set,
                                         submission=submission_set).head(10))

    _, dataset_loader = get_dataset_and_dataloader(
        train_set,
        features=config.FEATURES_EMBEDDING + ['next_city_id'],
        batch_size=config.BATCH_SIZE
    )
    dataset_batches_cuda = batches_to_device(dataset_loader)

    print_gpu_usage(0)

    # train model from best configuration
    model_configuration = get_base_configuration()
    model_hash = train_model_for_folds(dataset_batches_cuda,
                                       train_set,
                                       model_configuration,
                                       n_models=1)

    # get and save results table
    results, acc_dict = get_model_performance_data(test_set, de)
    filter_results_table(results)

    get_plot_from_accuracy(single=acc_dict[(model_hash, 'single')],
                           ensemble=acc_dict[(model_hash, 'ensemble')])

    # build ensemble with checkpoints of models identified by `model_hash`
    get_final_submission(submission_set, model_hash, de)
