This is a collection of code examples focusing on porting FORTRAN codes to run on the IBM OpenPOWER DOE platforms using OpenMP or CUDA.  The purpose of these is to provide both learning aids for developers and a regression test for compilers supporting OpenMP4.5 and CUDA FORTRAN, etc.

While these examples have a heavy FORTRAN emphasis, some examples also include C++ usage for use in demonstrating OpenMP or CUDA with mixed C++/FORTRAN language executables.

* examples - general examples on how to use OpenMP with more advanced concepts such as custom memory allocators, mapping function pointers, etc.
* iso_c_bindings - The OpenMP API for FORTRAN only provides a subset of the C API.  This directory contains iso_c_bindings of some of these missing API routines.
* macro_layer - An example of using a macro layer over OpenMP pragmas.
* partially_supported_constructs - Examples of OpenMP constructs that need enhancements to meet FORTRAN needs.
* testing - A collection of code examples that encountered issues during compiler testing, performance testing, and tool testing with CUDA and OpenMP on the DOE SIERRA platform using IBM XL OpenMP and CUDA FORTRAN.  May be useful as a test suite for compilers wanting to augment the very simple OpenMP examples available on the openmp.org website, especially in areas using FORTRAN derived or other features not well covered in those simpler examples.
* usability_issues - Examples highlighting general usability issues with OpenMP.

RELEASE
LLNL-CODE-769479
