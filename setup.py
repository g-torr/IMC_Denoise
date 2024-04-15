# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:34:57 2021

@author: penglu
"""

from __future__ import absolute_import
from setuptools import setup, find_packages

setup(name='IMC_Denoise',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "scikit-learn",
          "tensorflow==2.6.0",
          "keras==2.6.0",
          "protobuf==3.20.3"
          "tifffile"
        ]
      )
