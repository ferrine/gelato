from setuptools import setup, find_packages
import os

REQUIREMENTS = [
    # use requirements.txt
    #'theano',
    #'lasagne',
    #'pymc3'
]
REQUIREMENTS_DEV = [
    'pep8',
    'coverage',
    'nose'
]

if os.path.exists('README.md'):
    with open('README.md') as f:
        LONG_DESCRIPTION = f.read()
else:
    LONG_DESCRIPTION = ''

if __name__ == '__main__':
    setup(
        name='gelato',
        packages=find_packages(),
        description='Bayesian desert for Lasagne',
        long_description=LONG_DESCRIPTION,
        author='Maxim Kochurov',
        author_email='maxim.v.kochurov@gmail.com',
        download_url='https://github.com/ferrine/gelato',
        install_requires=REQUIREMENTS,
        tests_require=REQUIREMENTS_DEV
    )
