from codecs import open as codecs_open

from setuptools import setup, find_packages

# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='vision',
    version='0.0.2',
    packages=find_packages(),
    package_data={'vision': ['docs']},
    url='',
    license='',
    description='Item2Item recomender',
    long_description=long_description
)
