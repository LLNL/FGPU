! This example maps over an entire array to the GPU, then launches kernels to update sections of the array.
! It uses target update on those sections as each kernel finishes.
!
! This variant places the array inside a derived type.
program daxpy_array_slices_in_type
   use cudafor
   implicit none

   integer i, j, num_slices, num_values, a

   type daxpy_data
      real, pointer, contiguous, dimension(:,:) :: x, y
      integer :: a, b, c, d, e, f
      real :: g, h, i, j, k, l
   end type daxpy_data

   
   type(daxpy_data) :: the_data

   a = 5
   num_slices = 10
   num_values = 1024

   
   allocate(the_data%x(num_values,num_slices))
   allocate(the_data%y(num_values,num_slices))

   ! initialize arrays
   do j = 1, num_slices
      the_data%x(:,j) = DBLE(j)
      the_data%y(:,j) = DBLE(j)
   end do

   do j = 1, num_slices
      print *, "----------"
      print *, "before x(1:10,", j, ") ", the_data%x(1:10,j)
      print *, "----------"
   end do

   !$omp target enter data map(to:the_data)
   !$omp target enter data map(alloc:the_data%x)
   !$omp target enter data map(alloc:the_data%y)

   !$omp parallel do
   do j = 1, num_slices

      !$omp target update to(the_data%x(:,j), the_data%y(:,j))

      !$omp target teams distribute parallel do private(i) shared(a, the_data, j, num_values) default(none)
      do i = 1, num_values
         the_data%x(i,j) = a*the_data%x(i,j) + the_data%y(i,j)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j
      !$omp target update from(the_data%x(:,j), the_data%y(:,j))
   end do

   do j = 1, num_slices
      !$omp target exit data map(delete:the_data%x)
      !$omp target exit data map(delete:the_data%y)
      !$omp target exit data map(delete:the_data)
   end do

   do j = 1, num_slices
      print *, "----------"
      print *, "after x(1:10,", j, ") ", the_data%x(1:10,j)
      print *, "----------"
   end do

end program daxpy_array_slices_in_type
