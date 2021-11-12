# openMP is required here
# Two options that can be used to modify the chosen openMP flags
#set(BLT_OPENMP_COMPILE_FLAGS ""CACHE STRING "")
#set(BLT_OPENMP_LINK_FLAGS "" CACHE STRING "")

# add custom compiler flags for offloading 
blt_append_custom_compiler_flag(
  FLAGS_VAR FORTRAN_OPENMP_OFFLOAD
  DEFAULT  " "
  CLANG    " "
  GNU      " "
  INTEL    "-fiopenmp"
  CRAY     " "
  XL       "-qoffload"
  )

blt_append_custom_compiler_flag(
  FLAGS_VAR FORTRAN_USE_C_PREPROCESSOR
  DEFAULT  "-cpp"
  CLANG    "-cpp"
  GNU      "-cpp"
  INTEL    "-fpp"
  CRAY     "-eF"
  XL       " "
  )

blt_append_custom_compiler_flag(
  FLAGS_VAR FORTRAN_FREE_FORMAT
  DEFAULT  "-ffree-form"
  CLANG    "-ffree-form"
  GNU      "-ffree-form"
  INTEL    "-free"
  CRAY     "-ffree"
  XL       " "
  )
