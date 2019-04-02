from setuptools import setup, find_packages

__version__ = None
exec(open('Qsigma/version.py').read())

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='qsigma',
    version=__version__,
    packages=find_packages(exclude=(
        'dist', 'utils', 'build')),
    install_requires=reqs.strip().split('\n'),
    description="Q-sigma reinforcement learning algorithm",
    author='Ali Hejazizo',
    author_email='hejazizo@ualberta.ca',
    license='Apache 2.0',
    keywords="machine-learning reinforcement-learning",
    url="https://github.com/hejazizo",
)
