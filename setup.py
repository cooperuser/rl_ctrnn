# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="rl_ctrnn",
    version="0.1.0",
    description="CTRNN Biologically-inspired Reinforcement Learning",
    long_description=readme,
    author="Cooper b. Anderson",
    author_email="andersc7@rose-hulman.edu",
    url="https://github.com/cooperuser/rl_ctrnn",
    license=license,
    packages=find_packages(exclude=("tests", "test", "docs"))
)
