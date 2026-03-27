#!/usr/bin/env python3
"""
Setup script for Task-Guided Multi-Annotation Triplet Learning (TG-MATL)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="task-guided-matl",
    version="1.0.0",
    author="M. Zhou",
    description="Task-Guided Multi-Annotation Triplet Learning for Remote Sensing Representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meilunzhou/Task-Guided-MATL",
    packages=find_packages(exclude=["experiments", "tests", "*.tests"]),
    package_data={
        'matl': [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        "comet-ml>=3.30.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "umap-learn>=0.5.3",
    ],
)
