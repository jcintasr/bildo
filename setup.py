from setuptools import find_packages, setup

setup(
    name='bildo',
    packages=find_packages(include=['bildo']),
    version='0.0.1',
    descrìption='GDAL wrapper',
    author='jcintasr',
    setup_requires=["pytest-runner"],
    tests_require=['pytest'],
    test_suite="tests",
    license="GPLv3",
    )
