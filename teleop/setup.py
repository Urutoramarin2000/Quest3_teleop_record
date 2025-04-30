from setuptools import setup, find_packages

setup(
    name="teleop_cowa",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'zarr',
    ],
) 