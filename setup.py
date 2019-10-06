import os, sys
from setuptools import setup, find_packages

setup(
	name="eclib",
	version="0.0.1",
	install_requires=["numpy"],
	packages_dir={"","eclib"},
	packages=["eclib","eclib.base","eclib.benchmarks","eclib.operations","eclib.optimizers"]
)
