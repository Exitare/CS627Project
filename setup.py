#!/usr/bin/env python

from distutils.core import setup

with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='CS 627 Final Project',
      version='1.0',
      description='My final project',
      author='Exitare',
      author_email='exitare@exitare.de',
      url='https://exitare.de',
      install_requires=requirements,
      )
