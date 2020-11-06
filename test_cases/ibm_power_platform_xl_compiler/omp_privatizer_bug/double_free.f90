MODULE procedures
   IMPLICIT NONE

CONTAINS
  SUBROUTINE foo(Y)

   DOUBLE PRECISION, DIMENSION(:,:,:,:), INTENT(IN) :: Y

   INTEGER :: i,j,k
   DOUBLE PRECISION, DIMENSION(SIZE(Y,4)) :: Ys

   !$omp target teams distribute parallel do collapse(3) default(none) &
   !$omp& private(i,j,k,Ys) &
   !$omp& shared(Y)
   DO i=1,SIZE(Y,1)
    DO j=1,SIZE(Y,2)
     DO k=1,SIZE(Y,3)

      Ys = MIN(MAX(Y(i,j,k,:), 0.0D0), 1.0D0)
   
     END DO ! k
    END DO ! j
   END DO ! i

   !$omp end target teams distribute parallel do

  END SUBROUTINE
END MODULE procedures

PROGRAM fmain
   USE PROCEDURES
   IMPLICIT NONE

   DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:,:) :: Y

   ALLOCATE(Y(32,32,32,8))

   CALL foo(Y)

END PROGRAM fmain
