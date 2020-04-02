program collapse

  implicit none

  real, dimension(1000)   :: data1 = 1.0e0
  real, dimension(100,10) :: data2 = 1.0e0
  integer :: i,j

  !$omp parallel do simd
  do i=1,size(data1,1)
    data1(i) = data1(i) + 1.0e0
  end do
  !$omp end parallel do simd

  !$omp parallel do
  do i=1,size(data1,1)
    data1(i) = data1(i) + 1.0e0
  end do
  !$omp end parallel do
  
  !$omp parallel do simd collapse(2)
  do j=1,size(data2,2)
    do i=1,size(data2,1)
      data2(i,j) = data2(i,j) + 1.0e0
    end do
  end do
  !$omp end parallel do simd
  
  !$omp parallel do collapse(2)
  do j=1,size(data2,2)
    do i=1,size(data2,1)
      data2(i,j) = data2(i,j) + 1.0e0
    end do
  end do
  !$omp end parallel do

end program collapse
