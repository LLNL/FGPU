! This example attempts to map over a slice of an array, operate on it, then map it back.
program map_array_slices
   implicit none

   integer i, j, num_slices, num_values, a

   real, pointer, contiguous, dimension(:,:) :: x, y

   a = 5
   num_slices = 10
   num_values = 1024

   allocate(x(num_values,num_slices))
   allocate(y(num_values,num_slices))

   ! initialize arrays
   print *, "----------"
   do j = 1, num_slices
      x(:,j) = DBLE(j)
      y(:,j) = DBLE(j)
      print *, "before x(1:10,", j, ") ", x(1:10,j)
   end do
   print *, "----------"

!$omp parallel do
   do j = 1, num_slices

! Map and work on slice

      !$omp target teams distribute parallel do private(i) shared(a, x, y, j, num_values) map(tofrom:x(:,j)) map(to:y(:,j)) default(none)
      do i = 1, num_values
         x(i,j) = a*x(i,j) + y(i,j)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j

   end do
!$omp end parallel do

   print *, "----------"
   do j = 1, num_slices
      print *, "after x(1:10,", j, ") ", x(1:10,j)
   end do
   print *, "----------"

end program map_array_slices
