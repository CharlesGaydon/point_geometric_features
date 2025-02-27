# ----------------------------------------------------------------------#
#      setuptools setup script for compiling pgeof extensions           #
# ----------------------------------------------------------------------#
""" 
Loic Landrieu 2023
"""

from setuptools import setup, Extension
import numpy
import os

# targets and compile options
name = "pgeof"

include_dirs = [numpy.get_include(), "include"]  # find the Numpy headers

EIGEN_LIB_PATH = os.environ.get("EIGEN_LIB_PATH", None)

if EIGEN_LIB_PATH is not None:
    include_dirs.append(EIGEN_LIB_PATH)

print("includes")
print(include_dirs)

# Compilation and linkage options
if os.name == 'nt':  # windows
    extra_compile_args = ["/DPGEOF_WINDOWS"]
    extra_link_args = []
elif os.name == 'posix':  # linux
    extra_compile_args = ["-std=c++11", "-fopenmp"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not yet supported.')


#  Compilation
mod = Extension(
    name,
    # list source files
    ["./src/pgeof_cpy.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args)

setup(name=name, ext_modules=[mod])
