import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='microtubule_analysis',
    version='0.0.1',
    author='Alex Lettenberger',
    author_email='alex.lettenberger@gmail.com',
    description='Functions for calculating and analyzing microtubule catastrophe.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)