! This example maps over an entire array to the GPU, then launches kernels to update sections of the array.
! It uses target update on those sections as each kernel finishes.
program daxpy_array_slices
   implicit none

   integer i, j, num_slices, num_values, a
   real, pointer, contiguous, dimension(:,:) :: x, y

   a = 5
   num_slices = 10
   num_values = 1024

   allocate(x(num_values,num_slices))
   allocate(y(num_values,num_slices))

   ! initialize arrays
   do j = 1, num_slices
      x(:,j) = DBLE(j)
      y(:,j) = DBLE(j)
   end do

   do j = 1, num_slices
      print *, "----------"
      print *, "before x(1:10,", j, ") ", x(1:10,j)
      print *, "----------"
   end do

   !$omp target data map(alloc:x, y)

   !$omp parallel do
   do j = 1, num_slices

      !$omp target update to(x(:,j), y(:,j))

! NOTE - not sure the additional map on the teams line is needed.  Added it so
! the runtime knows that we've mapped over the array slice needed and to not
! perform any implicit map of x or y.
      !$omp target teams distribute parallel do private(i) shared(a, x, y, j, num_values) map(alloc:x(:,j), y(:,j)) default(none)
      do i = 1, num_values
         x(i,j) = a*x(i,j) + y(i,j)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j
      !$omp target update from(x(:,j), y(:,j))
   end do
   !$omp end parallel do

   !$omp end target data

   do j = 1, num_slices
      print *, "----------"
      print *, "after x(1:10,", j, ") ", x(1:10,j)
      print *, "----------"
   end do

end program daxpy_array_slices
