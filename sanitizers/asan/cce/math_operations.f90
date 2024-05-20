! This is the math_operations.f90 file
module math_operations
    implicit none

contains
    function add_numbers(a, b)
        implicit none
        real, intent(in) :: a, b
        real :: add_numbers

        add_numbers = a + b
    end function add_numbers
end module math_operations

