module nvtx_bindings_mod

use iso_c_binding
implicit none

! Provides seven colors so far.
integer, private :: nvtx_col(7) = [ Z'0000ff00', Z'000000ff', Z'00ffff00', &
                            Z'00ff00ff', Z'0000ffff', Z'00ff0000', Z'00ffffff']
character, private, target :: nvtx_tempName(256)

type, bind(C):: nvtxEventAttributes
  integer(C_INT16_T):: version=1
  integer(C_INT16_T):: size=48 !
  integer(C_INT):: category=0
  integer(C_INT):: colorType=1 ! NVTX_COLOR_ARGB = 1
  integer(C_INT):: color
  integer(C_INT):: payloadType=0 ! NVTX_PAYLOAD_UNKNOWN = 0
  integer(C_INT):: reserved0
  integer(C_INT64_T):: payload   ! union uint,int,double
  integer(C_INT):: messageType=1  ! NVTX_MESSAGE_TYPE_ASCII     = 1 
  type(C_PTR):: message  ! ascii char
end type

interface nvtxRangePushEx

  ! push range with custom label and custom color
  subroutine nvtxRangePushEx(event) bind(C, name='nvtxRangePushEx')
  use iso_c_binding
  import:: nvtxEventAttributes
  type(nvtxEventAttributes):: event
  end subroutine
end interface

contains

subroutine nvtxPushRange(name,id)
  character(kind=c_char,len=*) :: name
  integer:: id
  type(nvtxEventAttributes):: event
  character(kind=c_char,len=256) :: trimmed_name
  integer :: i
  
  trimmed_name=trim(name)//c_null_char

  ! move scalar trimmed_name into character array tempName
  do i=1,LEN(trim(name)) + 1
     nvtx_tempName(i) = trimmed_name(i:i)
  enddo

  event%color=nvtx_col(mod(id,7)+1)
  event%message=c_loc(nvtx_tempName)
  call nvtxRangePushEx(event)

end subroutine

subroutine nvtxPopRange
  interface nvtxRangePop
    subroutine nvtxRangePop() bind(C, name='nvtxRangePop')
    end subroutine
  end interface

  call nvtxRangePop()
end subroutine

end module nvtx_bindings_mod


module openmp_device_memory_routines

   use iso_c_binding

   implicit none
   private

   public :: omp_target_alloc, omp_target_free, omp_target_associate_ptr, omp_target_disassociate_ptr

   interface

      type(c_ptr) function omp_target_alloc( num_bytes, device_num ) bind ( c, name = 'omp_target_alloc' )
        use iso_c_binding
        implicit none

        integer(c_size_t), value :: num_bytes
        integer(c_int), value :: device_num
      end function omp_target_alloc

      subroutine omp_target_free( h_ptr, device_num ) bind ( c, name = 'omp_target_free' )
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr
        integer(c_int), value :: device_num
      end subroutine omp_target_free

      integer (c_int) function omp_target_associate_ptr( h_ptr, d_ptr, num_bytes, offset, device_num) bind(c, name='omp_target_associate_ptr')
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr, d_ptr
        integer(c_size_t), value :: num_bytes, offset
        integer(c_int), value :: device_num
      end function omp_target_associate_ptr

      integer (c_int) function omp_target_disassociate_ptr( h_ptr, device_num) bind(c, name='omp_target_disassociate_ptr')
        use iso_c_binding
        implicit none

        type(c_ptr), value :: h_ptr
        integer(c_int), value :: device_num
      end function omp_target_disassociate_ptr

   end interface
end module openmp_device_memory_routines


subroutine testsaxpy_omp45_f

  use iso_c_binding
  use omp_lib
!  use cudafor
  use openmp_device_memory_routines
  use nvtx_bindings_mod

  implicit none
  integer :: N = 100
  integer :: i, j, k, err
 
  real(c_double), pointer :: x(:,:,:), y(:,:,:)
  type(c_ptr) :: x_cptr, y_cptr
  integer(c_size_t) :: num_bytes, offset
  real(c_double) :: a = 2.0

  call nvtxPushRange("FORTRAN_KERNEL", 6)

  !$omp target enter data map(to:N,a)

  allocate( x(N,N,N) )
  allocate( y(N,N,N) )

  x = 1.0
  y = 2.0

  call omp_set_default_device(0)

  num_bytes = sizeof(a)*N*N*N
  offset = 0

  call nvtxPushRange("F_target_alloc", 1)
  x_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  y_cptr = omp_target_alloc(num_bytes, omp_get_default_device() )
  call nvtxPopRange()

  call nvtxPushRange("F_target_assoc_ptr", 2)
  err = omp_target_associate_ptr( C_LOC(x), x_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on x failed."
  endif

  err = omp_target_associate_ptr( C_LOC(y), y_cptr, num_bytes, offset, omp_get_default_device() )
  if (err /= 0) then
     print *, "Target associate on y failed."
  endif
  call nvtxPopRange()

  call nvtxPushRange("F_target_update_to", 3)
  !$omp target update to(x,y)
  call nvtxPopRange()
 
  ! Clear y on host
  y = 0.0 

  call nvtxPushRange("F_daxpy_kernel", 4)
  !$omp target teams distribute parallel do private(i,j,k) shared(y,a,x,N) default(none) collapse(3)
  do k=1,N
    do j=1,N
      do i=1,N
      y(i,j,k) = a*x(i,j,k) + y(i,j,k)
      end do
     end do
  end do
  !$omp end target teams distribute parallel do
  call nvtxPopRange()

  call nvtxPushRange("F_target_update_from", 5)
  !$omp target update from(y)
  call nvtxPopRange()

  call nvtxPushRange("F_disassoc_ptr", 6)
  err = omp_target_disassociate_ptr( C_LOC(x), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on x failed."
  endif

  err = omp_target_disassociate_ptr( C_LOC(y), omp_get_default_device() )
  if (err /= 0) then
     print *, "Target disassociate on y failed."
  endif
  call nvtxPopRange()

  call nvtxPushRange("F_target_free", 6)
  call omp_target_free( x_cptr, omp_get_default_device() )
  call omp_target_free( y_cptr, omp_get_default_device() )
  call nvtxPopRange()

  !$omp target exit data map(delete:N,a)

  write(*,*) "Ran FORTRAN OMP45 kernel. Max error: ", maxval(abs(y-4.0))

  call nvtxPopRange()
end subroutine testsaxpy_omp45_f
