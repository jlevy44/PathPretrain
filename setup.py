from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
PACKAGES=[ 
            ]

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

setup(name='pathpretrain',
      version='0.1',
      description='Simple setup train image models.',
      url='https://github.com/jlevy44/PathPretrain',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':['pathpretrain=pathpretrain.train_model:main']
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['pathpretrain'],
      install_requires=PACKAGES)
