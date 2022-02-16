#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clairvoyants",
    version="0.0.1",
    author="Amy Richardson Fricke",
    author_email="amy.r.fricke@gmail.com",
    description="An ensemble time series analysis and forecast package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amyrfricke/clairvoyants",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "clairvoyants"},
    #packages=setuptools.find_packages(where="clairvoyants"),
    packages=["clairvoyants"],
    python_requires=">=3.6",
)