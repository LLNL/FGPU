FUNCTION almost_equal(x, gold, tol) RESULT(b)
  implicit none
  REAL,  intent(in) :: x
  REAL,  intent(in) :: gold
  REAL,  intent(in) :: tol
  LOGICAL              :: b
  b = ( gold * (1 - tol)  <= x ).AND.( x <= gold * (1+tol) )
END FUNCTION almost_equal


! This example attempts to map over a slice of an array, operate on it, then map it back.
program map_array_slices
   implicit none

   integer i, j, num_slices, num_values, a

   LOGICAL :: almost_equal
   real, pointer, contiguous, dimension(:,:) :: x, y
   real :: expected_value;
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

   !!$omp parallel do
   do j = 1, num_slices
   ! Map and work on slice
      !$omp target teams distribute parallel do private(i) shared(a, x, y, j, num_values) map(tofrom:x(:,j)) map(to:y(:,j)) default(none)
      do i = 1, num_values
         x(i,j) = a*x(i,j) + y(i,j)
      end do
      !$omp end target teams distribute parallel do
   end do
   !!$omp end parallel do

   do j = 1, num_slices
     expected_value = a * DBLE(j) + DBLE(j)
     do i = 1, num_values
        IF ( .NOT.almost_equal( x(i,j),expected_value, 0.1) ) THEN
            print*,  "x(", i, ",", j, ")",  " | cur", x(i,j),  "ref", expected_value
            STOP 120
        ENDIF
     end do
   end do

end program map_array_slices
