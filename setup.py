import setuptools
from distutils.extension import Extension
import numpy
import sys

if sys.argv[-2] == "--cyth":
    USE_CYTHON = sys.argv[-1]
    sys.argv.pop()
    sys.argv.pop()
else:
    USE_CYTHON = False


ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("cyth_corr_window", ["PIV/cyth_corr_window" + ext],
                        include_dirs=[numpy.get_include()]),
              Extension("sym_filt", ["PIV/sym_filt" + ext]),
              Extension("ais_module", ["PIV/ais_module" + ext],
                        include_dirs=[numpy.get_include()])]


if USE_CYTHON:
    from Cython.Build import cythonize
    directives = {'linetrace': False,
                  'language_level': 3, 'boundscheck': False}
    extensions = cythonize(extensions, annotate=True,
                           compiler_directives=directives)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PIV",
    version="1.3",
    author="Matt Edwards",
    author_email="m.edwards@bristol.ac.uk",
    description="A framework for analysis PIV images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MattEdwards94/pivDevPython",
    install_requires=[
        'scipy==1.3.2',
        'numpy',
        'h5py',
        'pillow',
        'sklearn',
        'scikit-image',
        'matplotlib',
        'bottleneck',
    ],
    packages=setuptools.find_packages(),
    package_dir={'PIV': 'PIV'},
    package_data={'PIV': ['Data/*.csv']},
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
