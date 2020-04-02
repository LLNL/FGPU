program testmap

  use objects
  use ompdata
  

  ! Allocate the data
  call setup_objects(1)
  call point_to_objects(1)


  ! Set data on host
  prim_ptr%v3(1,1,1) = 9.99D0
  
  ! Map to device
  !$ call ompdata_prim_to()
   
  call setDeviceData(3.0D0)

  
  ! This should be 9.99 on host and 3 on device
  print*,prim_ptr%v3(1,1,1)

  !$omp target exit data map(from:prim_ptr%v1)
  print*,prim_ptr%v3(1,1,1)
  print*,prim_ptr%v1(1,1,1)
  !$omp target enter data map(to:prim_ptr%v1)


  !$ call ompdata_prim_from()
  print*,prim_ptr%v3(1,1,1)


end program testmap


subroutine setDeviceData(val)
  use objects
  real(kind=8), intent(in) :: val
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
