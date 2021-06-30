import setuptools
from PyTorchTrainer.version import __version__



setuptools.setup(
    name="PyTorchTrainer",
    version=__version__,
    author="Dat Tran",
    author_email="viebboy@gmail.com",
    description="Python utility for training pytorch models",
    long_description="Python utility for training pytorch models",
    long_description_content_type="text",
    url="https://github.com/viebboy/PyTorchTrainer",
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    classifiers=['Operating System :: POSIX', ],
    install_requires=['python_version >= "2.7"' or 'python_version >= "3.4"',
                      'torch >= 1.4.0',
                      'tqdm >= 4.36.1'],
    setup_requires=['torch >= 1.4.0',
                    'tqdm >= 4.36.1']
)
