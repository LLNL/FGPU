! This example attempts to map over a slice of an array, operate on it, then map it back.
! It will loop and operate over each slice of the array.
program daxpy_array_slices
   use cudafor
   implicit none

   integer i, j, num_slices, num_values, a

   real, pinned, allocatable, dimension(:,:) :: x, y

   a = 5
   num_slices = 80
   num_values = 1024

   allocate(x(num_values,num_slices))
   allocate(y(num_values,num_slices))

   ! initialize arrays
   do j = 1, num_slices
      x(:,j) = DBLE(j)
      y(:,j) = DBLE(j)
   end do

!$omp parallel do
   do j = 1, num_slices
      print *, "----------"
      print *, "before x(1:10,", j, ") ", x(1:10,j)
      print *, "before y(1:10,", j, ") ", y(1:10,j)
      print *, "----------"

! Map slice to GPU

      !$omp target enter data map(to:x(:,j))
      print *, "Mapped x(:,", j, ")"
      !$omp target enter data map(to:y(:,j))
      print *, "Mapped y(:,", j, ")"

! Work on slice

      !$omp target teams distribute parallel do private(i) shared(a, x, y, j, num_values) default(none)
      do i = 1, num_values
         x(i,j) = a*x(i,j) + y(i,j)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j

! Map slice back to CPU

      !$omp target exit data map(from:x(:,j))
      !$omp target exit data map(from:y(:,j))

      print *, "----------"
      print *, "after x(1:10,", j, ") ", x(1:10,j)
      print *, "after y(1:10,", j, ") ", y(1:10,j)
      print *, "----------"
   end do
!$omp end parallel do

end program daxpy_array_slices
