from setuptools import setup, find_packages


setup(
    name='pymatlib',
    version='0.1.0',  # Update this version as needed
    author='Rahil Doshi',  # Replace with your name
    author_email='rahil.doshi@fau.de',  # Replace with your email
    description='A Python based material library',
    long_description=open('README.md').read(),  # Ensure you have a README.md file
    long_description_content_type='text/markdown',
    url='https://i10git.cs.fau.de/rahil.doshi/pymatlib',  # Replace with your repository URL
    packages=find_packages(where='src'),  # Automatically find packages in the src directory
    package_dir={'': 'src'},  # Set the source directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',  # Adjust as necessary
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.10',  # Specify the minimum Python version required
    install_requires=[
        'numpy>=1.18.0',  # Specify required packages and their versions
        'sympy>=1.7.0',
        'pytest>=6.0.0',
        'pystencils@git+https://i10git.cs.fau.de/pycodegen/pystencils.git@v2.0-dev'
    ],
    extras_require={
        'dev': [
            'pytest-cov',  # For coverage reporting during tests
            'flake8',      # For style checking
            'black',       # For code formatting
        ],
    },
    include_package_data=True,  # Include package data specified in MANIFEST.in
)
