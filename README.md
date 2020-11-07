This is a collection of code examples focusing on porting FORTRAN codes to run DOE heterogenous architecture CPU+GPU machines.  The purpose of these is to provide both learning aids for developers and OpenMP and CUDA code examples for testing vendor compilers capabilities.

While these examples have a heavy FORTRAN emphasis, some examples also include C++ usage for use in demonstrating OpenMP or CUDA with mixed C++/FORTRAN language executables.

This set of examples can be built with CMake.  It requires BLT ( https://github.com/LLNL/blt ).

Some examples of compiler settings for platforms are in host_configs.

Quickstart:
git clone this repo into 'FGPU'
git clone BLT into 'BLT'
mkdir build

Assume you now have the directories:
FGPU
BLT
build
cd build

To build on an IBM Power platform
cmake ../FGPU -DBLT_ROOT=../BLT -C ../FGPU/host_configs/blueos.cmake
make


RELEASE
LLNL-CODE-769479
