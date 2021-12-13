from setuptools import setup
from setuptools import find_packages


setup(name='Facial Expression Detection',
      version='0.1.6',
      description='Detect facial expression from images',
      author='Team # - Projet Vision',
      author_email='abouba181@gmail.com',
      license='MIT',
      install_requires=['opencv-python', 'tensorflow', 'numpy'],
      packages=find_packages())
