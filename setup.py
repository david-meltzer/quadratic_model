from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
INSTALL_REQUIRES = [
    'torch>=1.13.1',
    'functorch>=1.13.1',
    'num2words'
]

setup(
    name='Quadratic Model',
    license='MIT License',
    author='David Meltzer',
    author_email='davidhmeltzer@gmail.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/david-meltzer/quadratic_model',
    long_description=long_description,
    packages=find_packages(),
    long_description_content_type='text/markdown',
    description='Catapult Dynamics',
    python_requires='>=3.6')
