! Compiling with IBM XLF with -O2 works.  Compiling with -O3 causes an incorrect answer.
FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  real(kind=8),  intent(in) :: x
  real(kind=8),  intent(in) :: gold
  real :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal

  module array_max
  USE iso_c_binding
  implicit none

! Holds an index location of an array, and value at that location
  TYPE loc_and_value_type
     REAL(c_double) :: val = -1.0D100
     INTEGER(c_int), DIMENSION(3) :: index = (/0,0,0/)
  END type loc_and_value_type

  !$omp declare reduction(mymaxloc: loc_and_value_type : omp_out = getmax(omp_in,omp_out) )

  contains

   FUNCTION iMAXLOC(var)
     IMPLICIT NONE
     DOUBLE PRECISION, DIMENSION(:,:,:), intent(in) :: var
     INTEGER, DIMENSION(3) :: iMAXLOC
     INTEGER :: ifunr,jfunr,kfunr
     TYPE(loc_and_value_type) :: max1,max2

     !$omp target teams distribute parallel do collapse(3) reduction(mymaxloc:max1) private(max2)
     do kfunr=1,size(var,3)
        do jfunr=1,size(var,2)
           do ifunr=1,size(var,1)
              max2%val = var(ifunr,jfunr,kfunr)
              max2%index = (/ifunr,jfunr,kfunr/)
              max1 = getmax(max2,max1)
           end do
        end do
     end do
     !$omp end target teams distribute parallel do
     iMAXLOC = max1%index
   END FUNCTION iMAXLOC

   FUNCTION getmax(max1,max2)
     IMPLICIT NONE
     !$omp declare target
     type(loc_and_value_type), INTENT(IN) :: max1,max2
     TYPE(loc_and_value_type) :: getmax
     if (max2%val > max1%val) then
        getmax%val   = max2%val
        getmax%index = max2%index
     else
        getmax%val   = max1%val
        getmax%index = max1%index        
     end if
   END FUNCTION getmax

   FUNCTION iMAXVAL(var)
     IMPLICIT NONE
     DOUBLE PRECISION, DIMENSION(:,:,:), intent(in) :: var
     DOUBLE PRECISION :: iMAXVAL
     INTEGER :: ifunr,jfunr,kfunr
     iMAXVAL = -1.0D100
     !$omp target teams distribute parallel do collapse(3) reduction(max:iMAXVAL)
     do kfunr=1,size(var,3)
        do jfunr=1,size(var,2)
           do ifunr=1,size(var,1)
              iMAXVAL = MAX(iMAXVAL, var(ifunr,jfunr,kfunr) )
           end do
        end do
     end do
     !$omp end target teams distribute parallel do     
   END FUNCTION iMAXVAL

  end module array_max

  program test_max
    USE iso_c_binding
    use array_max, only : imaxloc,imaxval
    implicit none

    real(kind=8), dimension(100,100,100) :: v,w
    real(kind=8) :: vmax=-100.0_8
    real(kind=8) :: vmax_gold=-100.0_8
    integer :: i,j,k,lmax(3)
    LOGICAL :: almost_equal

    ! Populate array with random numbers.
    call random_number(v)

    ! Init to zero
    w = 0.0_8

    ! Check locations and values on CPU using intrinsics.
    vmax_gold = maxval(v)
    lmax = maxloc(v)
    print *,'fortran intrinsics on CPU:'
    print *,'cpu max',vmax_gold,'at',lmax

    ! Check locations and values using custom openmp reduction function.
    vmax = -100.0_8
    lmax = -1

    !$omp target data map(to:v) map(alloc:w)

    !$omp target teams distribute parallel do collapse(3)
    do k=1,100 ; do j=1,100 ; do i=1,100
      w(i,j,k) = v(i,j,k)  ! on target
    end do ; end do ; end do
    !$omp end target teams distribute parallel do

#ifdef _OPENMP
    ! Zero out the host array version, make sure nothing is operating on this array.
    w = 0.0_8
#endif

    vmax = iMAXVAL(w)
    lmax = iMAXLOC(w)
    IF ( .NOT.almost_equal(vmax,vmax_gold, 0.1) ) THEN
      WRITE(*,*)  'Expected', vmax,  'Got', vmax_gold
      STOP 112
    ENDIF
    print *, "extrinsics on GPU, first call:"
    print *,'gpu max',vmax,'at',lmax

    vmax = iMAXVAL(w)
    lmax = iMAXLOC(w)
    IF ( .NOT.almost_equal(vmax,vmax_gold, 0.1) ) THEN
      WRITE(*,*)  'Expected', vmax,  'Got', vmax_gold
      STOP 112
    ENDIF
    print *, "extrinsics on GPU, second call:"
    print *,'gpu max',vmax,'at',lmax

    vmax = iMAXVAL(w)
    lmax = iMAXLOC(w)
    IF ( .NOT.almost_equal(vmax,vmax_gold, 0.1) ) THEN
      WRITE(*,*)  'Expected', vmax,  'Got', vmax_gold
      STOP 112
    ENDIF
    print *, "extrinsics on GPU, third call:"
    print *,'gpu max',vmax,'at',lmax

    !$omp end target data

  end program test_max

