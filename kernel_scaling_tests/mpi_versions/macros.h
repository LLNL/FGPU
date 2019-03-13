! Defines utility macros for use in Teton

! PROFILER CALLS
! Macro to start/stop a profiler range.
! 
! Currently, only calls to NVIDIA NVTX library are provided.
! Future profilers could be Google perftools, LLNL Caliper, etc.
#define START_RANGE(name, id) call nvtxStartRange(name, id)
#define END_RANGE() call nvtxEndRange()
