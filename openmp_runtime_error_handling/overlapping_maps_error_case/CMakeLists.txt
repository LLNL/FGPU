set(target "omp_runtime_reporting_overlapping_maps")

add_executable(${target}
               map_overlap.f90)
install(TARGETS ${target} DESTINATION bin)
add_test(test_${target} ${target})

message(STATUS "Setting omp_runtime_reporting_overlapping_maps test to expect a FAIL, will work on more later.")
set_tests_properties(test_${target} PROPERTIES WILL_FAIL TRUE)

# Some properties that can be set to control test pass/fail criteria:
#WILL_FAIL: If set to true, this will invert the pass/fail flag of the test.
#PASS_REGULAR_EXPRESSION: If set, the test output will be checked against the specified regular expressions and at least one of the regular expressions has to match, otherwise the test will fail.
#Example: PASS_REGULAR_EXPRESSION "TestPassed;All ok"
#FAIL_REGULAR_EXPRESSION: If set, if the output will match to one of specified regular expressions, the test will fail.a

