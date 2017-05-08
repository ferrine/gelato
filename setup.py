from setuptools import setup, find_packages
import codecs
import re
import os

here = os.path.abspath(os.path.dirname(__file__))

REQUIREMENTS = [
    # use requirements.txt
    'lasagne',
#    'pymc3'
]
REQUIREMENTS_DEV = [
    'pep8',
    'coverage',
    'nose'
]


try:
    with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
        LONG_DESCRIPTION = readme_file.read()

    with codecs.open(os.path.join(here, 'gelato', 'version.py'), encoding='utf-8') as version_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M)
        VERSION = version_match.group(1)
except Exception:
    LONG_DESCRIPTION = ''
    VERSION = ''

if __name__ == '__main__':
    setup(
        name='gelato',
        version=VERSION,
        packages=find_packages(),
        description='Bayesian dessert for Lasagne',
        long_description=LONG_DESCRIPTION,
        author='Maxim Kochurov',
        author_email='maxim.v.kochurov@gmail.com',
        download_url='https://github.com/ferrine/gelato',
        install_requires=REQUIREMENTS,
        tests_require=REQUIREMENTS_DEV
    )
