from distutils.core import setup
from Cython.Build import cythonize

directives = {'linetrace': False, 'language_level': 3}
setup(
    ext_modules=cythonize("sym_filt.pyx", annotate=True)
)
