#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinsor',
  version='0.1.0',
  description='You like tinygrad? You know einsum?',
  author='DKormann',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages = ['tinsor'],
  install_requires=["tinygrad",],
  python_requires='>=3.8',
  include_package_data=True)