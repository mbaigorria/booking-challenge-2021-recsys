import os
from os import listdir
from os.path import isfile
from typing import List

import recsys

PACKAGE_PATH = os.path.abspath(os.path.dirname(recsys.__file__))
RESOURCES_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, '..', 'resources'))


def get_resources_path(file_path: str) -> str:
    """
    Get resources path from file path.
    """
    path = os.path.join(RESOURCES_PATH, file_path)
    dirs = '/'.join(path.split('/')[:-1])

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    return path


def get_path(*args, dirs=None, format=None, filename=None, **kwargs) -> str:
    """
    Get path from args and kwargs.
    """
    path = []
    for arg in args:
        path.append(str(arg))
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                path.append(k)
        else:
            path.append('{}_{}'.format(k, v))

    dirs_str = ''
    if dirs is not None:
        if type(dirs) is not list:
            dirs = [dirs]
        dirs_str = '/'.join(dirs) + '/'

    path = get_resources_path(dirs_str + '_'.join(path))
    if filename is not None:
        path += filename
    if format is not None:
        path += "." + format

    return path


def get_model_ckpt_paths(model_hash: str, checkpoint_type='accuracy_at_k') -> List:
    """
    Get model checkpoints paths from `model_hash` by `checkpoint_type`.
    """
    base_path = get_path(f"models/{model_hash}")
    ckpt_paths = [f"{base_path}/{f}" for f in listdir(base_path) if isfile(f"{base_path}/{f}")]
    return sorted(list(filter(lambda s: checkpoint_type in s, ckpt_paths)))


def get_model_arch_path(model_hash) -> str:
    """
    Get model architecture paths from `model_hash`.
    """
    return get_path(dirs="architectures",
                    hash=model_hash,
                    filename=None,
                    format='json')
