import os
import setuptools

with open(os.path.join("README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dream_edit",
    version="0.0.1",
    author="Tiger Lab",
    description="Dream Edit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ltl3A87/DreamEdit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
    ],
)