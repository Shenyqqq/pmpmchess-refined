import setuptools
from setuptools import setup, Extension
import pybind11
import sys
import os


try:
    pybind11_include_dir = pybind11.get_include()
except AttributeError:
    print("Warning: pybind11.get_include() not found. Attempting to locate manually.")
    pybind11_path = os.path.dirname(pybind11.__file__)
    pybind11_include_dir = os.path.join(pybind11_path, 'include')
    if not os.path.isdir(pybind11_include_dir):
        print(f"Error: Could not find pybind11 include directory at {pybind11_include_dir}")
        sys.exit(1)


cpp_module = Extension(
    'katago_cpp_core',
    sources=[
        'game.cpp',
        'GameInterface.cpp',
        'MCTS.cpp',
        'pybind_wrapper.cpp'
    ],
    include_dirs=[
        pybind11_include_dir,
    ],
    language='c++',
    extra_compile_args=(
        ['/std:c++17', '/O2'] if sys.platform == 'win32' else ['-std=c++17', '-O2', '-Wall']
    ),
)

setup(
    name='katago_cpp_core',
    version='1.0',
    author='shenyqqq',
    description='C++ core for pmpmchess',
    ext_modules=[cpp_module],
    zip_safe=False,
)