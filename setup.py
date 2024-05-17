from setuptools import setup

setup(
    name='vehiclegym',
    version='1.0',
    packages=['data','vehiclegym','vehicle_models'],
    package_dir={'': 'src'},
    url='',
    license='MIT',
    author='rudolf',
    author_email='rudolf.reiter@imtek.uni-freiburg.de',
    description='Playground for automotive motion planning and control algorithms'
)
