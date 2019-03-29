PROGRAM fmain
   IMPLICIT NONE

   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: T
   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: U
   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: V
   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: W
   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: X
   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: Y
   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: Z

   INTEGER :: I,J,K,LEN1,LEN2,LEN3

   LEN1 = 128
   LEN2 = 128
   LEN3 = 128

   ALLOCATE(T(LEN1,LEN2,LEN3))
   ALLOCATE(U(LEN1,LEN2,LEN3))
   ALLOCATE(V(LEN1,LEN2,LEN3))
   ALLOCATE(W(LEN1,LEN2,LEN3))
   ALLOCATE(X(LEN1,LEN2,LEN3))
   ALLOCATE(Y(LEN1,LEN2,LEN3))
   ALLOCATE(Z(LEN1,LEN2,LEN3))

   T = 1.1
   U = 2.2
   V = 3.3
   W = 4.4
   X = 5.5
   Y = 6.6
   Z = 7.7

!----------------------------------------------------------------
   T = U*V*V
   W = X*Y*Y
   Z = Z*Z*Z

   print *, "CPU serial code kernel SUM(Z) is: ", SUM(Z)

!----------------------------------------------------------------
   T = 1.1
   W = 4.4
   Z = 7.7

   !$omp parallel workshare default(none) shared(T,U,V,W,X,Y,Z)
   T = U*V*V
   W = X*Y*Y
   Z = Z*Z*Z
   !$omp end parallel workshare

   print *, "CPU OpenMP workshare kernel SUM(Z) is: ", SUM(Z)

!----------------------------------------------------------------
   T = 1.1
   W = 4.4
   Z = 7.7

   !$omp target data map(to:T,U,V,X,Y) map(tofrom:Z)

   !$omp target
   !$omp parallel workshare default(none) shared(T,U,V,W,X,Y,Z)
   T = U*V*V
   W = X*Y*Y
   Z = Z*Z*Z
   !$omp end parallel workshare
   !$omp end target

   !$omp end target data

   print *, "GPU OpenMP workshare kernel SUM(Z) is: ", SUM(Z)
!----------------------------------------------------------------
   T = 1.1
   W = 4.4
   Z = 7.7

   !$omp target data map(to:T,U,V,X,Y) map(tofrom:Z)

   !$omp target teams
   !$omp parallel workshare default(none) shared(T,U,V,W,X,Y,Z)
   T = U*V*V
   W = X*Y*Y
   Z = Z*Z*Z
   !$omp end parallel workshare
   !$omp end target teams

   !$omp end target data

   print *, "GPU OpenMP teams workshare kernel SUM(Z) is: ", SUM(Z)
!----------------------------------------------------------------
   T = 1.1
   W = 4.4
   Z = 7.7

   !$omp parallel do collapse(3) default(none) &
   !$omp& shared(T,U,V,W,X,Y,Z,LEN1,LEN2,LEN3) &
   !$omp& private(I,J,K)
   DO I=1,LEN1
    DO J=1,LEN2
     DO K=1,LEN3
      T(I,J,K) = U(I,J,K) * V(I,J,K) * V(I,J,K)
      W(I,J,K) = X(I,J,K) * Y(I,J,K) * Y(I,J,K)
      Z(I,J,K) = Z(I,J,K) * Z(I,J,K) * Z(I,J,K)
     END DO
    END DO
   END DO      
   !$omp end parallel do
   print *, "CPU OpenMP explicit loop kernel SUM(Z) is: ", SUM(Z)

!----------------------------------------------------------------
   T = 1.1
   W = 4.4
   Z = 7.7

   !$omp target data map(to:T,U,V,X,Y) map(tofrom:Z)

   !$omp target parallel do collapse(3) default(none) &
   !$omp& shared(T,U,V,W,X,Y,Z,LEN1,LEN2,LEN3) &
   !$omp& private(I,J,K)
   DO I=1,LEN1
    DO J=1,LEN2
     DO K=1,LEN3
      T(I,J,K) = U(I,J,K) * V(I,J,K) * V(I,J,K)
      W(I,J,K) = X(I,J,K) * Y(I,J,K) * Y(I,J,K)
      Z(I,J,K) = Z(I,J,K) * Z(I,J,K) * Z(I,J,K)
     END DO
    END DO
   END DO      
   !$omp end target parallel do

   !$omp end target data

   print *, "GPU OpenMP explicit loop kernel SUM(Z) is: ", SUM(Z)
!----------------------------------------------------------------

   T = 1.1
   W = 4.4
   Z = 7.7

   !$omp target data map(to:T,U,V,X,Y) map(tofrom:Z)

   !$omp target teams distribute parallel do collapse(3) default(none) &
   !$omp& shared(T,U,V,W,X,Y,Z,LEN1,LEN2,LEN3) &
   !$omp& private(I,J,K)
   DO I=1,LEN1
    DO J=1,LEN2
     DO K=1,LEN3
      T(I,J,K) = U(I,J,K) * V(I,J,K) * V(I,J,K)
      W(I,J,K) = X(I,J,K) * Y(I,J,K) * Y(I,J,K)
      Z(I,J,K) = Z(I,J,K) * Z(I,J,K) * Z(I,J,K)
     END DO
    END DO
   END DO      
   !$omp end target teams distribute parallel do

   !$omp end target data

   print *, "GPU OpenMP teams explicit loop kernel SUM(Z) is: ", SUM(Z)
!----------------------------------------------------------------


END PROGRAM fmain
