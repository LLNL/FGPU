! This example maps over an entire array to the GPU, then launches kernels to update sections of the array.
! It uses target update on those sections as each kernel finishes.
!
! This variant places the array inside a derived type.
! This example maps over an entire array to the GPU, then launches kernels to update sections of the array.
! It uses target update on those sections as each kernel finishes.
FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  REAL,  intent(in) :: x
  REAL,  intent(in) :: gold
  REAL,  intent(in) :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal

program daxpy_array_slices_in_type
   implicit none

   integer i, j, num_slices, num_values, a
   real :: expected_value
   LOGICAL :: almost_equal

   type daxpy_data
      real, pointer, contiguous, dimension(:,:) :: x, y
      !Not user just to have more fun
      integer :: a = 1
      real :: g = 1.0
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

   
   !$omp target data map(to:the_data) map(alloc:the_data%x, the_data%y)

   !$omp parallel do
   do j = 1, num_slices

      !$omp target update to(the_data%x(:,j), the_data%y(:,j))

! NOTE - not sure the additional map on the teams line is needed.  Added it so
! the runtime knows that we've mapped over the array slice needed and to not
! perform any implicit map of x or y.
!      !$omp target teams distribute parallel do private(i) shared(a, the_data, j, num_values) map(alloc:the_data%x(:,j), the_data%y(:,j)) default(none)
     !$omp target teams distribute parallel do private(i) shared(a, the_data, j, num_values) default(none)
      do i = 1, num_values
         the_data%x(i,j) = a*the_data%x(i,j) + the_data%y(i,j)
      end do
      !$omp end target teams distribute parallel do

      !$omp target update from(the_data%x(:,j), the_data%y(:,j))
   end do
   !$omp end parallel do

   !$omp end target data

   do j = 1, num_slices
     expected_value = a * DBLE(j) + DBLE(j)
     do i = 1, num_values
        IF ( .NOT.almost_equal( the_data%x(i,j),expected_value, 0.1) ) THEN
            print*,  "x(", i, ",", j, ")",  " | cur", the_data%x(i,j),  "ref", expected_value
            STOP 120
        ENDIF
     end do
   end do

end program daxpy_array_slices_in_type
