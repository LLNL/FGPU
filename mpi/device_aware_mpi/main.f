PROGRAM gpudirect

	use setup, only : setup_types, remove_types, op_ptr
	use mpi
	use iso_c_binding
  
  implicit none
	
	double precision, dimension(:,:), allocatable :: arr, send, recv
	double precision, dimension(:,:), pointer :: send_ptr, recv_ptr
  integer :: rank,np,mpierr,request	! MPI stuff
  integer mpistatus(MPI_STATUS_SIZE)
  integer :: neigh_l, neigh_r	! Communication stuff
  integer :: a,nx,ny,i,j,sim		! Other
  character(len=32) :: arg

!-----------------------------------------------------------------------------------------------
  ! Setup
  
  ! MPI
  CALL MPI_INIT(mpierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD,rank,mpierr)

  ! Print troop status
  IF ( rank== 0 ) THEN
  	print *,'Welcome to the GPU-Direct testing center! \n Calling all MPI ranks:'
  ENDIF
  print *,'Rank #',rank,' reporting for duty!'
  
  ! Get more MPI info
  call MPI_COMM_SIZE(MPI_COMM_WORLD,np,mpierr)

  ! Saddling horses
  neigh_l = rank-1
  neigh_r = rank+1
  
  if(neigh_l==-1) neigh_l = np-1
  if(neigh_r==np) neigh_r = 0
  
  !if (rank==1) print *,'Rank',rank,': Left neighbor = ',neigh_l,'. Should be ',rank-1
  print *,'Rank',rank,': Left neighbor = ',neigh_l
  !if (rank==1) print *,'Rank',rank,': Right neighbor = ',neigh_r,'. Should be ',rank+1
  print *,'Rank',rank,': Right neighbor = ',neigh_r

	! Define constants
  a = 100
  sim =1		! Default
  
  ! Read program input
  call getarg(1,arg); read(arg,'(I10)') sim

  ! Select case to run:
  ! 1: Allocating temporary arrays that are passed through MPI call, using only CPU 
  ! 2: Allocating temporary arrays that are passed through MPI call, using GPU-direct (currently how Miranda works)
  ! 3: Using scratch space that is allocated before hand as part of derived type, CPU only
  ! 4: Using scratch space that is allocated before hand as part of derived type, using GPU-direct (DOES NOT WORK, 9/5/2019)
  ! 5: Passing a pointer through the MPI call that references the scratch array inside a derived type; CPU only
  ! 6: Passing a pointer through the MPI call that references the scratch array inside a derived type; using GPU-direct (Works but severely impacts performance)
  ! 7: Sending a subsection of an array through an MPI call, CPU only.
  ! 8: Sending a subsection of an allocatated array through an MPI call using GPU-direct (DOES NOT WORK, 9/5/2019)
  
  
  select case( sim )
  
!-----------------------------------------------------------------------------------------------
  ! Using allocatable arrays for MPI - no GPU offload
  case(1)
  
  ! Allocate and initialize arrays
  call setup_types(a,rank)
  if(rank==0) print *,'==============Using allocatable temporary arrays for MPI on CPU============'
	call MPI_Barrier(MPI_COMM_WORLD,mpierr)
	
	! Allocate temporary MPI buffers
  allocate( send(a,a), recv(a,a) )

  ! Fill MPI buffers
  do i=1,a; do j=1,a
  	send(i,j) = op_ptr%mult(1)%array(i,j)
  	recv(i,j) = -1
  end do; end do

  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  !print *,'Rank #',rank,', send initialized with value: ',send(1,1)
  !print *,'Rank #',rank,', recv initialized with value: ',recv(1,1)

  ! Send your value of array to your right neighbor and receive your left neighbor's value of array
  call MPI_SendRecv( send, size(send), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 recv, size(recv), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  									
  ! Update value of op_ptr%mult(1)%array
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = recv(i,j)
  end do; end do
  
  ! Check solution
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l 
  
  call remove_types()
  deallocate(send,recv)
  
  
!-----------------------------------------------------------------------------------------------
  ! Using allocatable arrays for MPI - GPU version
  case(2)
  
  ! Allocate and initialize arrays
  call setup_types(a,rank)
  if(rank==0) print *,'==============Using allocatable temporary arrays for MPI w/GPU============'
	call MPI_Barrier(MPI_COMM_WORLD,mpierr)
	
	! Allocate temporary MPI buffers
  allocate( send(a,a), recv(a,a) )

  ! Allocate data on GPU
  !$omp target data map(alloc:send,recv)
  ! Fill MPI buffers on GPU
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	send(i,j) = op_ptr%mult(1)%array(i,j)
  	recv(i,j) = -1
  end do; end do
  !$omp end target teams distribute parallel do

  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  !print *,'Rank #',rank,', send initialized with value: ',send(1,1)
  !print *,'Rank #',rank,', recv initialized with value: ',recv(1,1)
 
  ! Send your value of array to your right neighbor and receive your left neighbor's value of array
  !$omp target data use_device_ptr(send,recv)
  call MPI_SendRecv( send, size(send), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 recv, size(recv), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  !$omp end target data
  									
  ! Update value of op_ptr%mult(1)%array
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = recv(i,j)
  end do; end do
  !$omp end target teams distribute parallel do
  
  ! Force a memcpy to host for printing out solution
	!$omp target update from(op_ptr%mult(1)%array)
  ! Check solution
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l 
  
  ! End target data region and copy memory back to host
  !$omp end target data
  call remove_types()
  deallocate(send,recv)
  
 
!-----------------------------------------------------------------------------------------------
  ! Using Derived Type arrays, no GPU offload
  case(3)
  
  call setup_types(a,rank)
  
  if(rank==0) print *,'==============Using derived type scratch arrays for MPI on CPU============'
  call MPI_Barrier(MPI_COMM_WORLD,mpierr)
  
  ! Fill temp arrays
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%send(i,j) = op_ptr%mult(1)%array(i,j)
  	op_ptr%mult(1)%recv(i,j) = -1
  end do; end do
  
  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  
  !! Send your value of array to your right neighbor and receive your left neighbor's value of array
  call MPI_SendRecv( op_ptr%mult(1)%send, size(op_ptr%mult(1)%send), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 op_ptr%mult(1)%recv, size(op_ptr%mult(1)%recv), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  
  ! Update value of array 
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = op_ptr%mult(1)%recv(i,j)
  end do; end do
  
  ! Check solution
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l 
  
  call remove_types()
  
  
!-----------------------------------------------------------------------------------------------
  ! Using Derived Type buffer arrays through MPI with GPU offload !!! Does not give correct answer, 9/5/2019
  case(4)
  
  call setup_types(a,rank)		! Allocates array,send,recv as derived type components and initializes array
  
  if(rank==0) print *,'==============Using derived type scratch arrays for MPI w/GPU============'
  call MPI_Barrier(MPI_COMM_WORLD,mpierr)
  
  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  
  ! Fill temp arrays on GPU
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%send(i,j) = op_ptr%mult(1)%array(i,j)
  	op_ptr%mult(1)%recv(i,j) = -1
  end do; end do
  !$omp end target teams distribute parallel do
  
  ! Send your value of array to your right neighbor and receive your left neighbor's value of array
  !$omp target data use_device_ptr(op_ptr%mult(1)%send, op_ptr%mult(1)%recv)
  call MPI_SendRecv( op_ptr%mult(1)%send, size(op_ptr%mult(1)%send), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 op_ptr%mult(1)%recv, size(op_ptr%mult(1)%recv), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  !$omp end target data
  
  ! Update value of array 
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = op_ptr%mult(1)%recv(i,j)
  end do; end do
  !$omp end target teams distribute parallel do
  
  ! Force a memcpy to host for printing out solution
	!$omp target update from(op_ptr%mult(1)%array)
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l 
  
  call remove_types()
  
!-----------------------------------------------------------------------------------------------
  ! Using Derived Type arrays, no GPU offload, using pointers to arrays
  case(5)
  
  call setup_types(a,rank)
  
  if(rank==0) print *,'==============Using derived type scratch arrays but CPU pointers in MPI============'
  call MPI_Barrier(MPI_COMM_WORLD,mpierr)
  
  ! Associate pointers with buffer arrays
  send_ptr=>op_ptr%mult(1)%send
  recv_ptr=>op_ptr%mult(1)%recv
  
  ! Fill temp arrays
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%send(i,j) = op_ptr%mult(1)%array(i,j)
  	op_ptr%mult(1)%recv(i,j) = -1
  end do; end do
  
  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  
  !! Send your value of array to your right neighbor and receive your left neighbor's value of array
  call MPI_SendRecv( send_ptr, size(op_ptr%mult(1)%send), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 recv_ptr, size(op_ptr%mult(1)%recv), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  
  ! Update value of array 
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = op_ptr%mult(1)%recv(i,j)
  end do; end do
  
  ! Check solution
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l 
  
  call remove_types()
   
!-----------------------------------------------------------------------------------------------
  ! Using Derived Type buffer arrays through MPI with GPU offload and aliased pointers
  case(6)
  
  call setup_types(a,rank)		! Allocates array,send,recv as derived type components and initializes array
  
  if(rank==0) print *,'==============Using derived type scratch arrays for MPI w/GPU, with aliased pointer============'
  
  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  
  !$omp target data use_device_ptr(op_ptr%mult(1)%send,op_ptr%mult(1)%recv)
  send_ptr=>op_ptr%mult(1)%send
  recv_ptr=>op_ptr%mult(1)%recv
  !$omp end target data
  
  ! Fill temp arrays on GPU
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%send(i,j) = op_ptr%mult(1)%array(i,j)
  	op_ptr%mult(1)%recv(i,j) = -1
  end do; end do
  !$omp end target teams distribute parallel do
  
  ! Send your value of array to your right neighbor and receive your left neighbor's value of array
  !$omp target data use_device_ptr(send_ptr, recv_ptr)
  call MPI_SendRecv( send_ptr, size(op_ptr%mult(1)%send), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 recv_ptr, size(op_ptr%mult(1)%recv), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  !$omp end target data

  ! Update value of array 
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = op_ptr%mult(1)%recv(i,j)
  end do; end do
  !$omp end target teams distribute parallel do
  
  ! Force a memcpy to host for printing out solution
	!$omp target update from(op_ptr%mult(1)%array)
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l 
  
  call remove_types()  

!-----------------------------------------------------------------------------------------------
  ! Using allocatable arrays with subsections for MPI, CPU only
  case(7)
  
  ! Allocate and initialize arrays
  call setup_types(a,rank)
  
  if(rank==0) print *,'==============Using sub-sections of allocatable, temporary arrays for MPI on CPU============'
	call MPI_Barrier(MPI_COMM_WORLD,mpierr)

	! Allocate temporary MPI buffers
  allocate( send(a,a), recv(a,a) )

  ! Fill MPI buffers
  do i=1,a; do j=1,a
  	send(i,j) = op_ptr%mult(1)%array(i,j)
  	recv(i,j) = -1
  end do; end do

  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  !print *,'Rank #',rank,', send initialized with value: ',send(1,1)
  !print *,'Rank #',rank,', recv initialized with value: ',recv(1,1)

  ! Send your value of array to your right neighbor and receive your left neighbor's value of array
  call MPI_SendRecv( send(1:10,:), size(send(1:10,:)), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 recv(1:10,:), size(recv(1:10,:)), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  									
  ! Update value of op_ptr%mult(1)%array
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = recv(i,j)
  end do; end do
  
  ! Check solution
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array(1,1) after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array(11,1) after SendRecv: ',op_ptr%mult(1)%array(11,1),'. Should be', -1
  
  call remove_types()
  deallocate(send,recv)

!-----------------------------------------------------------------------------------------------
  ! Using allocatable arrays with subsections for MPI - GPU version -- doesn't work, segfaults on MPI command!
  case(8)
  
  ! Allocate and initialize arrays
  call setup_types(a,rank)
  if(rank==0) print *,'==============Using sub-sections of allocatable, temporary arrays for MPI w/GPU============'
	call MPI_Barrier(MPI_COMM_WORLD,mpierr)

	! Allocate temporary MPI buffers
  allocate( send(a,a), recv(a,a) )

  ! Allocate data on GPU
  !$omp target data map(alloc:send,recv)
  ! Fill MPI buffers on GPU
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	send(i,j) = op_ptr%mult(1)%array(i,j)
  	recv(i,j) = -1
  end do; end do
  !$omp end target teams distribute parallel do

  print *,'Rank #',rank,', op_ptr%mult(1)%array initialized with value: ',op_ptr%mult(1)%array(1,1),'. Should be', rank 
  !print *,'Rank #',rank,', send initialized with value: ',send(1,1)
  !print *,'Rank #',rank,', recv initialized with value: ',recv(1,1)
 
  ! Send your value of array to your right neighbor and receive your left neighbor's value of array
  !$omp target data use_device_ptr(send(1:10,:),recv(1:10,:))
  call MPI_SendRecv( send(1:10,:), size(send(1:10,:)), MPI_DOUBLE_PRECISION, neigh_r, 0, &
  									 recv(1:10,:), size(recv(1:10,:)), MPI_DOUBLE_PRECISION, neigh_l, 0, &
  									 MPI_COMM_WORLD, mpistatus, mpierr)
  !$omp end target data
  									
  ! Update value of op_ptr%mult(1)%array
  !$omp target teams distribute parallel do collapse(2)
  do i=1,a; do j=1,a
  	op_ptr%mult(1)%array(i,j) = recv(i,j)
  end do; end do
  !$omp end target teams distribute parallel do
  
  ! Force a memcpy to host for printing out solution
	!$omp target update from(op_ptr%mult(1)%array)
  ! Check solution
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array(1,1) after SendRecv: ',op_ptr%mult(1)%array(1,1),'. Should be', neigh_l
  print *,'Rank #',rank,', value of op_ptr%mult(1)%array(11,1) after SendRecv: ',op_ptr%mult(1)%array(11,1),'. Should be', -1
  
  ! End target data region and copy memory back to host
  !$omp end target data
  call remove_types()
  deallocate(send,recv)
  
  end select
  
  CALL MPI_FINALIZE(mpierr)
  
END PROGRAM gpudirect
