import pandas as pd

from common import config


def load_dataset(path=config.TRAIN_CSV):
    return pd.read_csv(path)


def load_holdout(path):
    return pd.read_csv(path)
