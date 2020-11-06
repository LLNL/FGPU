program mc1
   implicit none

   integer, parameter :: nsets = 1  ! number of 'sets'
   integer, parameter :: nmc = 1  ! number of trials in set
   integer, parameter :: n = 1       ! sample size in trial

   real, dimension(nsets,nmc) :: mu      ! result of each trial
   real :: random_numbers(nmc, n)        ! bunch of random numbers, since GPU doesn't have random_number subroutine...
   real :: mean, stdev                   ! mean and standard deviation
   integer :: k, j, i
   real :: y, u


   do j = 1, nmc
      do i = 1, n
         call random_number(u)            ! draw from Uniform(0,1)
         random_numbers(j, i) = u
      end do
   end do

   ! have to initialize again in kernel, firstprivate doesn't work in OMP45 GPU yet.
   y = 0.d0

!$omp target data map(to:random_numbers) map(from:mu)

!$omp target teams distribute num_teams(nsets) default(none) shared(random_numbers, mu, y, u) private(k)
   do k = 1, nsets
!$omp parallel do default(none) shared(random_numbers, mu) private(u,k,y)
! firstprivate(y)
      do j = 1, nmc
         y = 0.d0
         do i = 1, n
!            u = random_numbers(j, i)
!            y = y + u                        ! sum the draws
         end do
!         mu(k,j) = y / dble(n)                     ! return the sample mean

      end do
!$omp end parallel do
   end do
!$omp end target teams distribute

!$omp end target data

   do k = 1, nsets
      mean = sum(mu) / dble(nmc)
      stdev = sqrt( sum( (mu(k,:) - mean)**2 ) ) / dble(nmc)

      print *, 'mean(', k, ')',mean
      print *, 'stdev(', k, ')',stdev
   end do

end program mc1
