from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
with open("requirements.txt") as f:
      PACKAGES=f.read().splitlines()

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

setup(name='pathpretrain',
      version='0.1.3',
      description='Simple setup train image models.',
      url='https://github.com/jlevy44/PathPretrain',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=[],
      entry_points={
            'console_scripts':['pathpretrain-train=pathpretrain.train_model:main',
                               'pathpretrain-predict=pathpretrain.predict:main',
                               'pathpretrain-embed=pathpretrain.embed:main',
                               'pathpretrain-preprocess=pathpretrain.preprocess:main'
                               ]
      },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['pathpretrain'],
      install_requires=PACKAGES,
      extras_requires=dict(stain_norm="histomicstk==1.2.10".split()))
