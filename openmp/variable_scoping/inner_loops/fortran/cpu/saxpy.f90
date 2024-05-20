program saxpy
    use omp_lib
    implicit none

    integer, parameter :: n = 100000
    real(kind=4), dimension(n) :: x, y
    real(kind=4) :: a
    integer :: i, j, k
    real(kind=4) :: temp

    ! Initialize the scalar and vectors
    a = 2.0
    x = 1.0
    y = 2.0

! QUESTION
! Remove the 'private(j,k)'.  Compiler should complain these are not scoped due to the 'default(none)'.
! XL - does not catch the error, unlike the C++ example which does.  Is Fortran expected behavior different?
! Are these scoped as firstprivate in Fortran, but not C++?
! TEST
! Change the 'private(j,k)' to a shared(j,k).  The compiler should complain that loop iteration variables can not be shared.
! XL - does catch this error.

    ! Perform the SAXPY operation in parallel
    !$omp parallel do default(none) shared(x, y, a) private(i, j, k, temp)
    do i = 1, n
        temp = a * x(i)
        
        ! First inner serial loop
        do j = 1, 5
            temp = temp + j
        end do
        
        ! Second inner serial loop
        do k = 1, 3
            temp = temp - k
        end do
        
        y(i) = temp + y(i)
    end do
    !$omp end parallel do

    ! Print the first 10 results
    do i = 1, 10
        print *, y(i)
    end do

end program saxpy

