FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  DOUBLE PRECISION,  intent(in) :: x
  DOUBLE PRECISION,  intent(in) :: gold
  REAL,  intent(in) :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal

program testmap

  use objects
  use ompdata
  implicit none

  LOGICAL :: almost_equal
  LOGICAL :: r

  ! Allocate the data
  call setup_objects(1)
  call point_to_objects(1)

  ! Set data on host
  prim_ptr%v3(1,1,1) = 9.99D0
  
  ! Map to device
  !$ call ompdata_prim_to()
  
  call setDeviceData(3.0D0)

  
  ! This should be 9.99 on host and 3 on device
#ifdef _OPENMP
  r = almost_equal(prim_ptr%v3(1,1,1), 9.99D0, 0.1)  
#else
  r = almost_equal(prim_ptr%v3(1,1,1), 3.0D0, 0.1)
#endif
  IF ( .NOT.r) THEN
    WRITE(*,*)  '1/ Wrong value for prim_ptr%v3(1,1,1)', prim_ptr%v3(1,1,1)
    STOP 112
  ENDIF

  !$omp target
  prim_ptr%v1(1,1,1) = 1.0D0
  !$omp end target

  !$omp target exit data map(from:prim_ptr%v1)

  !V1 shoud have the corret value
  IF (.NOT.almost_equal(prim_ptr%v1(1,1,1),1.0D0,0.1)) THEN
    WRITE(*,*)  '2/ Wrong value for prim_ptr%v1(1,1,1)', prim_ptr%v1(1,1,1)
    STOP 112
  ENDIF

  !V3 didn't change
#ifdef _OPENMP
  r = almost_equal(prim_ptr%v3(1,1,1), 9.99D0, 0.1)
#else
  r = almost_equal(prim_ptr%v3(1,1,1), 3.0D0, 0.1)
#endif
  IF ( .NOT.r) THEN
    WRITE(*,*)  '3/ Wrong value for prim_ptr%v3(1,1,1)', prim_ptr%v3(1,1,1)
    STOP 112
  ENDIF

  !$omp target enter data map(to:prim_ptr%v1)
  !$ call ompdata_prim_from()

  IF (.NOT.almost_equal(prim_ptr%v3(1,1,1), 3.d0,0.1)) THEN
    WRITE(*,*)  '4/ Wrong value for prim_ptr%v3(1,1,1)', prim_ptr%v3(1,1,1)
    STOP 112
  ENDIF

end program testmap


subroutine setDeviceData(val)
  use objects
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(in) :: val
  integer :: i,j,k

  !$omp target teams distribute parallel do collapse(3)
  do i=1,10
     do j=1,10
        do k=1,10
           prim_ptr%v3(i,j,k) = val
        end do
     end do
  end do
  !$omp end target teams distribute parallel do

end subroutine setDeviceData
