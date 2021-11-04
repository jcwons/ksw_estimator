from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
                  Extension("bispectrum",["bispectrum.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=['-fopenmp -c=-DUSE_XSIMD -c=-march=native'],
    extra_link_args=['-fopenmp'],
    )
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(ext_modules = ext_modules, cmdclass = {'build_ext': build_ext},)

