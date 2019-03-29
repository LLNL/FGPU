DEBUGCLANG=-v -g -O0 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
OPTCLANG= -v -g -O0 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda

DEBUGXL=-v -g -qcuda -qsmp=noopt -qoffload -O0 -qfullpath -qnoinline -Xptxas -O0 -Xllvm2ptx -nvvm-compile-options=-opt=0
OPTXL= -v -O3 -qcuda -qsmp=omp -qoffload

# Grabbed these from examining what xlc and xlf does at final link with '-v'.  Probably overkill, can trim these down later.
XLLIBS=\
-L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib \
-L/usr/tce/packages/xl/xl-2019.02.07/xlC/16.1.1/lib \
-L/usr/tce/packages/xl/xl-2019.02.07/xlsmp/5.1.1/lib \
-L/usr/tce/packages/xl/xl-2019.02.07/xlmass/9.1.1/lib \
-L/usr/tce/packages/xl/xl-2019.02.07/lib \
-L/usr/tce/packages/cuda/cuda-9.2.148/targets/ppc64le-linux/lib \
-lxlcufdevice -lxldevice -lomptarget-nvptx -lcudadevrt -lcuda -lcudart -lxlf90_r -lxlcuf -lxlfmath -lxlcuf -lxlcuf_nonshr -lxldevice -lxlsmp -lxlopt -lxl -lxlfmath
# -libmc++ -lm

CUDALIBS=-L$(CUDA_HOME)/lib64 -lcudart

# Current link errors.
#saxpy_omp45_f.o: In function `__sti____cudaRegisterAll_18_Fsaxpy_omp45_f_f90_testsaxpy_omp45_f_':
#/usr/workspace/wsrzc/black27/git/code_examples_gpu/mixed_language_trials/xl_clang_omp_test/saxpy_omp45_f.fatbinreg.h:18: undefined reference to `__cudaRegisterLinkedBinary_18_Fsaxpy_omp45_f_f90_testsaxpy_omp45_f_'
#saxpy_omp45_f.o:(.toc+0x10): undefined reference to `__omptgt__img_start_nvptx64sm_35-nvidia-linux'
#saxpy_omp45_f.o:(.toc+0x18): undefined reference to `__omptgt__img_end_nvptx64sm_35-nvidia-linux'
#saxpy_omp45_f.o:(.toc+0x20): undefined reference to `__omptgt__host_entries_begin'
#saxpy_omp45_f.o:(.toc+0x28): undefined reference to `__omptgt__host_entries_end'
#saxpy_omp45_f.o:(.data+0x0): undefined reference to `__omptgt__img_start_nvptx64sm_35-nvidia-linux'
#saxpy_omp45_f.o:(.data+0x8): undefined reference to `__omptgt__img_end_nvptx64sm_35-nvidia-linux'
#saxpy_omp45_f.o:(.data+0x10): undefined reference to `__omptgt__host_entries_begin'
#saxpy_omp45_f.o:(.data+0x18): undefined reference to `__omptgt__host_entries_end'
#saxpy_omp45_f.o:(.data+0x30): undefined reference to `__omptgt__host_entries_begin'
#saxpy_omp45_f.o:(.data+0x38): undefined reference to `__omptgt__host_entries_end'

# Pure clang, works fine
debugclang:
	clang++  $(DEBUGCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	clang++ -DOMP45C $(DEBUGCLANG)  -c main.cc
	clang++ $(DEBUGCLANG) -o test_clang main.o saxpy_omp45_c.o

# Pure clang, works fine
optclang:
	clang++  $(OPTCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	clang++ -DOMP45C $(OPTCLANG)  -c main.cc
	clang++ $(OPTCLANG) -o test_clang main.o saxpy_omp45_c.o

# Try out a xlC OpenMP4.5 kernel and clang link, 
debugc:
	xlC_r  $(DEBUGXL) -qtgtarch=compute_70 -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	clang++ -DOMP45C $(DEBUGCLANG)  -c main.cc
	clang++ $(XLLIBS) $(DEBUGCLANG) -o test_clang main.o saxpy_omp45_c.o

# Add FORTRAN to the mix
debug:
	clang++  $(DEBUGCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(DEBUGXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	clang++ -DOMP45C -DOMP45F $(DEBUGCLANG)  -c main.cc
	clang++ $(DEBUGCLANG) -o test_clang main.o saxpy_omp45_c.o saxpy_omp45_f.o $(XLLIBS) $(CUDALIBS)

# Add FORTRAN to the mix
opt:
	clang++  $(OPTCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(OPTXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	clang++ -DOMP45C -DOMP45F $(OPTCLANG)  -c main.cc
	clang++ $(OPTCLANG) -o test_clang main.o saxpy_omp45_c.o saxpy_omp45_f.o $(XLLIBS) $(CUDALIBS)




# Reference test - all XL, works fine
debugxl:
	xlC_r  $(DEBUGXL) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(DEBUGXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -DOMP45C -DOMP45F $(DEBUGXL)  -c main.cc
	xlC_r $(DEBUGXL) -o test_xl main.o saxpy_omp45_c.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

# Reference test - all XL, works fine
optxl:
	xlC_r  $(OPTXL) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(OPTXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -DOMP45C -DOMP45F $(OPTXL)  -c main.cc
	xlC_r $(OPTXL) -o test_xl main.o saxpy_omp45_c.o saxpy_omp45_f.o $(XLLIBS)

# Reference test - all XL, link with XLF instead of XLC, works fine
optxlf:
	xlC_r  $(OPTXL) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(OPTXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	xlC_r -DOMP45C -DOMP45F $(OPTXL)  -c main.cc
	xlf_r -v $(OPTXL) -o test_xl main.o saxpy_omp45_c.o saxpy_omp45_f.o -L/usr/tce/packages/xl/xl-2019.02.07/xlf/16.1.1/lib -lxlf90_r -lxlcuf -lxlfmath

# EXPERIMENTAL
# Try out a clang compile and xlC link.  Currently hit errors:
#nvlink error   : Size doesn't match for 'omptarget_nvptx_device_State' in '/usr/tce/packages/xl/xl-2019.02.07/xlsmp/5.1.1/lib/libomptarget-nvptx.a:omptarget-nvptx_generated_omptarget-nvptx.cu.o', first specified in 'saxpy_omp45_c.o'
#nvlink error   : Size doesn't match for 'omptarget_nvptx_device_State' in '/usr/tce/packages/xl/xl-2019.02.07/xlsmp/5.1.1/lib/libomptarget-nvptx.a:omptarget-nvptx_generated_omp_data.cu.o', first specified in 'saxpy_omp45_c.o'
#nvlink error   : Undefined reference to 'omptarget_nvptx_device_simpleState' in 'saxpy_omp45_c.o'
#nvlink error   : Undefined reference to 'omptarget_nvptx_simpleThreadPrivateContext' in 'saxpy_omp45_c.o'
#nvlink error   : Undefined reference to 'usedSlotIdx' in 'saxpy_omp45_c.o'
debugmix1:
	clang++ $(DEBUGCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(DEBUGXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	clang++ $(DEBUGCLANG) -DOMP45F -DOMP45C -c main.cc
	xlf_r -L/usr/tce/packages/clang/clang-upstream-2018.12.03/ibm/omprt/lib -lxlsmp -l omptarget$(DEBUGXL) $(XLLIBS) -o test_mix main.o saxpy_omp45_c.o saxpy_omp45_f.o

# Try out XL for any source with OpenMP4.5, CLANG for source with only CPU OpenMP, and xlC link.
# This works fine since CLANG is only compiled CPU OpenMP.
mix2:
	xlC_r  $(DEBUGXL) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f $(DEBUGXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	clang++ -DOMP45C -DOMP45F -fopenmp  -c main.cc
	xlf_r $(DEBUGXL) -o test_mix main.o saxpy_omp45_c.o saxpy_omp45_f.o $(XLLIBS)

# Use custom libs directory to force both compilers to use same runtime libraries.
mix3:
	clang++ -L./libs $(DEBUGCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f -L./libs $(DEBUGXL) -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	clang++ -L./libs $(DEBUGCLANG) -DOMP45F -DOMP45C -c main.cc
	xlf_r -L./libs $(DEBUGXL) $(XLLIBS) -o test_mix main.o saxpy_omp45_c.o saxpy_omp45_f.o

mix4:
	clang++ -L./libs -lomptarget -lomptarget-nvptx $(OPTCLANG) -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	clang++ -L./libs -lomptarget -lomptarget-nvptx -DOMP45C $(OPTCLANG)  -c main.cc
	clang++ -L./libs -lomptarget -lomptarget-nvptx $(OPTCLANG) -o test_clang main.o saxpy_omp45_c.o

mix5:
	clang++  -fopenmp -c saxpy_omp45_c.cc -o saxpy_omp45_c.o
	xlf_r -d -qextname=testsaxpy_omp45_f -qsmp=omp -c saxpy_omp45_f.F90 -o saxpy_omp45_f.o
	clang++ -DOMP45C -DOMP45F -fopenmp  -c main.cc
	clang++ -fopenmp -o test_clang main.o saxpy_omp45_c.o saxpy_omp45_f.o $(XLLIBS) $(CUDALIBS)

clean:
	rm -f *.o  *.f90
	rm -f *.mod
	rm -f test_*
