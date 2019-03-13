program fmain
use derived_type_mod

implicit none

allocate(dt_ptr)
call dt_ctor(dt_ptr)

print *, "CPU dt_ptr%arr2(1,1):", dt_ptr%arr2(1,1)
print *, "CPU dt_ptr%arr3(1,1,1):", dt_ptr%arr3(1,1,1)

!$omp target update to(dt_ptr%arr2, dt_ptr%arr3)

!$omp target
call foo()
!$omp end target

!$omp target update from(dt_ptr%arr2, dt_ptr%arr3)

print *, "CPU dt_ptr%arr2(2,2):", dt_ptr%arr2(2,2)
print *, "CPU dt_ptr%arr3(2,2,2):", dt_ptr%arr3(2,2,2)

deallocate(dt_ptr)

end program fmain

subroutine foo()
use derived_type_mod
implicit none

!$omp declare target

write (*,*) "GPU dt_ptr%arr(1,1):", dt_ptr%arr2(1,1)
write (*,*) "GPU dt_ptr%arr3(1,1,1):", dt_ptr%arr3(1,1,1)

dt_ptr%arr2(2,2) = 1.0
dt_ptr%arr3(2,2,2) = 1.0

write (*,*) "GPU dt_ptr%arr(2,2):", dt_ptr%arr2(2,2)
write (*,*) "GPU dt_ptr%arr3(2,2,2):", dt_ptr%arr3(2,2,2)

return

end subroutine
