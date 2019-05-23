! This example attempts to map over slices of an array to a GPU, then launch kernels that act on those slices.

program daxpy_array_slices
   use cudafor
   implicit none

   integer i, j, num_slices, num_values, a

   real, pinned, allocatable, dimension(:,:) :: x, y

   a = 5
   num_slices = 2
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
      print *, "before y(1:10,", j, ") ", y(1:10,j)
      print *, "----------"
   end do

   do j = 1, num_slices
      !$omp target enter data map(to:x(:,j))
      print *, "Mapped x(:,", j, ")"
      !$omp target enter data map(to:y(:,j))
      print *, "Mapped y(:,", j, ")"
   end do

   do j = 1, num_slices

      !$omp target teams distribute parallel do private(i) shared(a, x, y, j, num_values) default(none)
      do i = 1, num_values
         x(i,j) = a*x(i,j) + y(i,j)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j
   end do

   do j = 1, num_slices
      !$omp target exit data map(from:x(:,j))
      !$omp target exit data map(from:y(:,j))
   end do

   do j = 1, num_slices
      print *, "----------"
      print *, "after x(1:10,", j, ") ", x(1:10,j)
      print *, "after y(1:10,", j, ") ", y(1:10,j)
      print *, "----------"
   end do

end program daxpy_array_slices
