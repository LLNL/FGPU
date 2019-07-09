! These macros provide a mechanism for enabling/disabling OpenMP4.5 target pragmas lines, along with
! inserting additional behavior before/after the pragmas.
!
! ENABLE_OMP_MACRO - enables any lines annotated with the OMP macro, should be set by the build system.
! ENABLE_OMP_TRACE_PRINTS - enables/disables trace prints before/after the annotated OpenMP lines.
!
! OMP - executes the OpenMP line, if macro enabled
! OMP1 - executes the OpenMP line and prepends optional behavior
! OMP2 - executes the OpenMP line and appends optional behavior
!
! If optional behavior is not enabled, OMP1 and OMP2 simply execute the OpenMP line.

! Uncomment to enable tracing.
#define ENABLE_OMP
#define ENABLE_OMP_TRACE_PRINTS
#define ENABLE_OMP_TEAMS_DISTRIBUTE
#define ENABLE_OMP_TARGET

! Try to mitigate the performance impact of the target construct on CPU runtime.
! Even if offloading disabled in compiler options, the presence of the targe
! tends to slow down CPU threaded code.
#if defined(ENABLE_OMP_TARGET)
#  define OMP_TARGET target
#else
#  define OMP_TARGET
#endif

! Try to mitigate the performance impact of teams distribute constructs on CPU runtime.
! Even if offloading disabled in compiler options, the presence of the teams distribute
! tends to slow down CPU threaded code.
#if defined(ENABLE_OMP_TEAMS_DISTRIBUTE)
#  define OMP_TEAMS_DISTRIBUTE teams distribute
#else
#  define OMP_TEAMS_DISTRIBUTE
#endif

#if defined(ENABLE_OMP)
#  define OMP(source) !$omp source

#  if defined(ENABLE_OMP_TRACE_PRINTS)
#    define OMP1(source) print *, "Line: source", " File: ", __FILE__ , " Line: ",__LINE__ ; !$omp source
#    define OMP2(source) !$omp source ; print *, "Line: source", " File: ", __FILE__ , " Line: ",__LINE__
#  else
#    define OMP1(source) !$omp source
#    define OMP2(source) !$omp source
#  endif

#else

#  define OMP(source)
#  define OMP1(source)
#  define OMP2(source)

#endif
