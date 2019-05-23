program daxpy_type_ptrs
   use cudafor
   implicit none

   integer i, j, num_items, num_arr_values, a

   type daxpy_data
      real, pointer, dimension(:) :: x
      real, pointer, dimension(:) :: y
      integer :: a, b, c, d, e, f
      real :: g, h, i, j, k, l
   end type daxpy_data

   type(daxpy_data), allocatable, dimension(:) :: data_ptrs

   num_items=10
   num_arr_values = 1024

   allocate(data_ptrs(num_items))

   do j = 1, num_items
      allocate(data_ptrs(j)%x(num_arr_values))
      allocate(data_ptrs(j)%y(num_arr_values))

      data_ptrs(j)%a = 5
      data_ptrs(j)%x(:) = DBLE(j)
      data_ptrs(j)%y(:) = DBLE(j)
   end do

!$omp parallel do
   do j = 1, num_items
      print *, "x(1:10):", data_ptrs(j)%x(1:10)
      print *, "y(1:10):", data_ptrs(j)%x(1:10)

      !$omp target enter data map(to:data_ptrs(j))
      !$omp target enter data map(to:data_ptrs(j)%x)
      !$omp target enter data map(to:data_ptrs(j)%y)
      print *, "Mapped data_ptrs(", j, ")"

      !$omp target teams distribute parallel do private(i) shared(data_ptrs, j, num_arr_values) default(none)
      do i = 1, num_arr_values
         data_ptrs(j)%x(i) = data_ptrs(j)%a*data_ptrs(j)%x(i) + data_ptrs(j)%y(i)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on data_ptrs(", j, ")"

      !$omp target exit data map(from:data_ptrs(j)%x)
      !$omp target exit data map(from:data_ptrs(j)%y)
      !$omp target exit data map(from:data_ptrs(j))

      print *, "x(1:10):", data_ptrs(j)%x(1:10)
      print *, "y(1:10):", data_ptrs(j)%x(1:10)
   end do
!$omp end parallel do

end program daxpy_type_ptrs
