! This example attempts to map over a slice of an array, operate on it, then map it back.
! It will loop and operate over each slice of the array.
!
! The XL OpenMP runtime currently issues errors when attempting to map over
! multiple slices of an array on the GPU at a time, issuing an error about
! overlapping ambiguous maps.
!
! This is likely due to the dope vector getting mapped multiple times.
! As a workaround, this version adds an array of pointers pointing to each
! slice of the array, and maps each of those.  This ensures each has it's own
! dope vector.
program daxpy_array_slices
   use cudafor
   implicit none

! You can't create an array of pointers in FORTRAN.  This is the closest thing.
   type ptr_wrapper
      real, dimension(:), pointer :: p_x => NULL()
      real, dimension(:), pointer :: p_y => NULL()
   end type ptr_wrapper

   integer i, j, num_slices, num_values, a

   type(ptr_wrapper), pinned, allocatable, dimension(:) :: slice_ptrs
   real, pinned, allocatable, target, dimension(:,:) :: x, y

   a = 5
   num_slices = 5
   num_values = 1024

   allocate(x(num_values,num_slices))
   allocate(y(num_values,num_slices))
   allocate(slice_ptrs(num_slices))

   ! initialize arrays, set up pointers to slices
   do j = 1, num_slices
      x(:,j) = DBLE(j)
      y(:,j) = DBLE(j)
      slice_ptrs(j)%p_x => x(:,j)
      slice_ptrs(j)%p_y => y(:,j)
   end do

! Pre-map over entire array of slice pointers
!$omp target enter data map(to:slice_ptrs)

!$omp parallel do
   do j = 1, num_slices
      print *, "----------"
      print *, "before x(1:10,", j, ") ", x(1:10,j)
      print *, "before y(1:10,", j, ") ", y(1:10,j)
      print *, "----------"

! Map slice to GPU

      !$omp target enter data map(to:slice_ptrs(j)%p_x)
      print *, "Mapped x slice: ", j
      !$omp target enter data map(to:slice_ptrs(j)%p_y)
      print *, "Mapped y slice: ", j

! Work on slice

      !$omp target teams distribute parallel do private(i) shared(slice_ptrs, a, x, y, j, num_values) default(none)
      do i = 1, num_values
         slice_ptrs(j)%p_x(i) = a*slice_ptrs(j)%p_x(i) + slice_ptrs(j)%p_y(i)
      end do
      !$omp end target teams distribute parallel do

      print *, "Ran daxpy on slice ", j

! Map slice back to CPU

      !$omp target exit data map(from:slice_ptrs(j)%p_x)
      !$omp target exit data map(delete:slice_ptrs(j)%p_y)

      print *, "----------"
      print *, "after x(1:10,", j, ") ", x(1:10,j)
      print *, "after y(1:10,", j, ") ", y(1:10,j)
      print *, "----------"
   end do
!$omp end parallel do

!$omp target exit data map(delete:slice_ptrs)

end program daxpy_array_slices
