from setuptools import setup, find_packages, Extension
# import pybind11


# Define the extension module
'''ext_modules = [
    Extension(
        "pymatlib.core.cpp.fast_interpolation",  # Module name in Python
        [
            "src/pymatlib/core/cpp/module.cpp",
            "src/pymatlib/core/cpp/binary_search_interpolation.cpp",
            "src/pymatlib/core/cpp/double_lookup_interpolation.cpp",
        ],
        include_dirs=[pybind11.get_include(),
                      "src/pymatlib/core/cpp/include"],
        extra_compile_args=['-O3', '-std=c++11'],  # Enable high optimization and C++11
        language='c++'
    ),
]'''

setup(
    name='pymatlib',
    version='0.1.0',  # Update this version as needed
    author='Rahil Doshi',
    author_email='rahil.doshi@fau.de',
    description='A Python based material library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://i10git.cs.fau.de/rahil.doshi/pymatlib',
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
        'pystencils@git+https://i10git.cs.fau.de/pycodegen/pystencils.git@v2.0-dev',
        # 'pybind11>=2.6.0',
        'pwlf>=2.5.1',
    ],
    extras_require={
        'dev': [
            'pytest-cov',  # For coverage reporting during tests
            'flake8',      # For style checking
            'black',       # For code formatting
        ],
    },
    # ext_modules=ext_modules,
    include_package_data=True,  # Include package data specified in MANIFEST.in
)
