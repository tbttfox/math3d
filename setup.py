import os

from setuptools import setup, find_packages

name = 'math3d'
dirname = os.path.dirname(os.path.abspath(__file__))

# Get the long description from the README file.
description="A math library for working in 3d graphics based on subclassing numpy arrays"

setup(
	name='{}'.format(name),
	version='0.1.0',
	description=description,
	long_description=description,
	url='https://github.com/tbttfox/{}'.format(name),

	license='MIT',
	classifiers=[
			'Development Status :: 4 - Beta',
			'Intended Audience :: Developers',
			'Programming Language :: Python',
			'Programming Language :: Python :: 2.7',
			'Programming Language :: Python :: 3',
			'Operating System :: OS Independent',
			"Topic :: Multimedia :: Graphics",
			"Topic :: Scientific/Engineering :: Mathematics",
			"Topic :: Scientific/Engineering :: Visualization"
			'License :: OSI Approved :: MIT License',
	],
	keywords=["3d", "math", "quaternion", "numpy"],
	packages=find_packages(exclude=['tests']),
	include_package_data=True,
	author='Tyler Fox <tbttfox@gmail.com>',
	install_requires=[],
	author_email='tbttfox@gmail.com>',
)
