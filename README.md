This is a collection of code examples focusing on porting FORTRAN codes to run DOE heterogenous architecture CPU+GPU machines.  The purpose of these is to provide both learning aids for developers and OpenMP and CUDA code examples for testing vendor compilers capabilities.

While these examples have a heavy FORTRAN emphasis, some examples also include C++ usage for use in demonstrating OpenMP or CUDA with mixed C++/FORTRAN language executables.

The code examples are currently being ported to CMake and CTest.
Some examples of setting CMake options for platforms are in the 'cmake' directory.

Quickstart:
git clone this repo into 'FGPU'
mkdir build

Assume you now have the directories:
FGPU
build

To build on an IBM Power platform
cd build
cmake ../FGPU -C ../FGPU/cmake/blueos.cmake
make
make test

RELEASE
LLNL-CODE-769479
