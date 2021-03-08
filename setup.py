# coding=utf-8
from setuptools import setup, find_packages

setup(
    name="Booking.com Sequence Aware Recommender System",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.1.5',
        'numpy>=1.16.2',
        'sklearn',
        'torch',
        'tqdm>=4.56.0',
        'GPUtil>=1.4.0',
        'psutil>=5.8.0',
        'humanize>=3.2.0',
        'matplotlib>=3.1.3',
        'memoized_property>=1.0.3',
        'seaborn>=0.11.1'
    ],
    author="Martín Baigorria Alonso",
    author_email="martinbaigorria@gmail.com",
    description="Solution for Booking.com next destination recommendation challenge"
)
