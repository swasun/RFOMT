from setuptools import find_packages, setup

setup(
    name='bolsonaro',
    packages=find_packages(where="code", exclude=['doc', 'dev']),
    package_dir={'': "code"},
    version='0.1.0',
    description='Bolsonaro project of QARMA non-permanents: deforesting random forest using OMP.',
    author='QARMA team',
    license='MIT',
)
