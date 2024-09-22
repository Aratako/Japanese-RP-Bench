# data.py

from datasets import load_dataset


def load_dataset_wrapper(dataset_repo: str, split: str, cache_dir: str):
    return load_dataset(dataset_repo, split=split, cache_dir=cache_dir)
