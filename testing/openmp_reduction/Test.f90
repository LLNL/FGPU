program Main

        implicit none
        integer :: i,j,N,k,count
        double precision :: temp

        N=1000

        do count = 1, 100

        !$omp parallel do private(k,temp) shared(N) default(none)
        do k = 1,100
        
                temp = 0d0

                !$OMP target teams distribute parallel do private(i,j) shared(N) collapse(2) reduction(+:temp) default(none)
                do j=1,N
                        do i=1,N
                                temp = temp + 1d-5
                        enddo
                enddo
                !$OMP end target teams distribute parallel do

                print *, "k:", k, "; temp = ", temp

        enddo
        !$omp end parallel do

        print *, "--------------"

        enddo
end program Main
