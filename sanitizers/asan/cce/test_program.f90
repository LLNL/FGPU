! This is the test_program.f90 file
program test_program
    use math_operations
    implicit none
    real :: num1, num2, result

    ! Initialize variables
    num1 = 10.0
    num2 = 5.0

    ! Call the function from the module
    result = add_numbers(num1, num2)

    ! Print the result
    print *, 'The sum of ', num1, ' and ', num2, ' is ', result
end program test_program

