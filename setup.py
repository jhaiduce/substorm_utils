"""Config for PyPI."""

from setuptools import find_packages
from setuptools import setup

# List of dependencies installed via `pip install -e .`
# by virtue of the Setuptools `install_requires` value below.
requires = [
    'numpy',
    'scipy',
    'spacepy',
    'matplotlib',
    'pytz',
    'cdaweb@git+https://github.com/jhaiduce/cdaweb',
    'backports.functools_lru_cache',
    'cache_decorator@git+https://github.com/jhaiduce/cache_decorator'
]

setup(
    author='John Haiducek',
    author_email='jhaiduce@umich.edu',
    version='0.0.1',
    zip_safe=True,
    packages=find_packages(),
    name='substorm_utils',
    install_requires=requires
)
