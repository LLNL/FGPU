! This example maps over an entire array to the GPU, then launches kernels to update sections of the array.
! It uses target update on those sections as each kernel finishes.
!
! This variant places the array inside a derived type.
program map_array_slices_in_type
   implicit none

   integer i, j, num_slices, num_values, a

   type daxpy_data
      real, pointer, contiguous, dimension(:,:) :: x, y
      integer :: a, b, c, d, e, f = 1
      real :: g, h, i, j, k, l = 1.0
   end type daxpy_data
   
   type(daxpy_data), pointer :: the_data

   a = 5
   num_slices = 10
   num_values = 1024
   
   allocate(the_data)
   allocate(the_data%x(num_values,num_slices))
   allocate(the_data%y(num_values,num_slices))

   ! initialize arrays
   print *, "----------"
   do j = 1, num_slices
      the_data%x(:,j) = DBLE(j)
      the_data%y(:,j) = DBLE(j)
      print *, "before x(1:10,", j, ") ", the_data%x(1:10,j)
   end do
   print *, "----------"

   !$omp target data map(to:the_data)

   !$omp parallel do
   do j = 1, num_slices

! Map and work on slice

      !$omp target teams distribute parallel do private(i) shared(a, the_data, j, num_values) map(tofrom:the_data%x(:,j)) map(to:the_data%y(:,j)) default(none)
      do i = 1, num_values
         the_data%x(i,j) = a*the_data%x(i,j) + the_data%y(i,j)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j

   end do
!$omp end parallel do

!$omp end target data 
   print *, "----------"
   do j = 1, num_slices
      print *, "after x(1:10,", j, ") ", the_data%x(1:10,j)
   end do
   print *, "----------"

end program map_array_slices_in_type
