from setuptools import setup
from setuptools import find_packages

setup(name='rllib',
      version='0.1',
      description='Less idiotic RLlib trainers',
      author='Max Pumperla',
      install_requires=[],
      extras_require={
          'tests': []
      },
      packages=find_packages(),
      license='MIT',
      zip_safe=False,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Environment :: Console',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3'
      ])