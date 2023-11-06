from setuptools import find_packages, setup

import pathlib

# Commands: 
# python setup.py sdist
# twine upload dist/*
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Dependencies required to use
INSTALL_REQS = [
    "torch == 1.13.1",
    "pytorch_lightning == 1.9.1",
    # "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl",
    "spacy==3.5.0",
    "transformers==4.26.1",
    "pyserini==0.20.0",
    "matplotlib==3.6.3",
    "flair",
    "nltk",
    "pandas",
    "tqdm",
    "accelerate",
]

setup(
    name="qa-natver",
    version="1.00",
    description="Repository for 'Question Answering for Natural Logic-based Fact Extraction', EMNLP2023.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Raldir/QA-NatVer",
    author="Rami Aly",
    author_email="rmya2@cam.ac.uk",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=INSTALL_REQS,
    python_requires=">=3.8",
)