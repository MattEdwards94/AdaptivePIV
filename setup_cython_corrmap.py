from distutils.core import setup
from Cython.Build import cythonize
import numpy

directives = {'linetrace': False, 'language_level': 3}
setup(
    ext_modules=cythonize("cyth_corr_window.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
