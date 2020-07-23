program test
   implicit none
   
   integer, allocatable, dimension(:) :: node1
   integer, allocatable, dimension(:) :: node2
   integer, allocatable, dimension(:) :: num_field_values_arr

   real, allocatable, dimension(:,:) :: A
   real, allocatable, dimension(:,:) :: B
   real, allocatable, dimension(:,:,:) :: C
   real, allocatable, dimension(:,:,:) :: D
   real, allocatable, dimension(:) :: foo

   integer :: node, val, zone
   integer :: num_zones
   integer :: num_nodes_zone1
   integer :: num_nodes_zone2
   integer :: num_nodes_total
   integer :: num_dimensions
   integer :: num_field_values

! NOTE - In this code example we have a case with two 3d zones.
! - One with 8 nodes (hex)
! - One with with 5 nodes (pyramid)
! - Each node has a certain number of field values to compute.


   num_zones = 2
   num_nodes_zone1 = 8
   num_nodes_zone2 = 5
   num_nodes_total = num_nodes_zone1 + num_nodes_zone2
   num_dimensions = 3
   num_field_values = 10

   ! Array holding number of field values per zone
   allocate(num_field_values_arr(num_zones))
   num_field_values_arr(1)= 8
   num_field_values_arr(2)= 10

   ! Arrays holding indicies to first and last corner for each zone.
   allocate(node1(num_zones))
   node1(1) = 1
   node1(2) = num_nodes_zone1 + 1

   allocate(node2(num_zones))
   node2(1) = num_nodes_zone1
   node2(2) = num_nodes_zone1 + num_nodes_zone2

   allocate(A(num_field_values, num_nodes_total))
   allocate(B(num_field_values, num_nodes_total))
   allocate(C(num_dimensions,num_field_values, num_nodes_total))
   allocate(D(num_dimensions,num_field_values, num_nodes_total))
   allocate(foo(num_dimensions))

   ! Some dummy data
   A = 1.0
   B = 2.0
   C = 3.0
   D = 4.0
   foo = 5.0

   print *, SHAPE(node1), " node1: ", node1
   print *, SHAPE(node2), " node2: ", node2

   do zone= 1, num_zones
       print *, "Zone: ", zone
 
! Check that indirection in outer loop is fine
!$omp target teams distribute parallel do collapse(2) &
!$omp& default(none) shared(node1,node2,zone,A,B,C,D,foo,num_field_values) private(node, val)
       do node= node1(zone), node2(zone)
         do val=1, num_field_values
           A(val,node) = DOT_PRODUCT( foo(:), C(:,val,node) )
           B(val,node) = DOT_PRODUCT( foo(:), D(:,val,node) )
         enddo
       enddo
!$omp end target teams distribute parallel do

     enddo

   do zone= 1, num_zones
       print *, "Zone: ", zone
! Check that indirection in inner loop is fine too.  (we would usually reverse
! the layout of the array if we had this looping pattern, but this is adequate
! to verify the loop works)
!$omp target teams distribute parallel do collapse(2) &
!$omp& default(none) shared(node1,node2,zone,A,B,C,D,foo,num_field_values) private(node, val)
       do val=1, num_field_values
         do node= node1(zone), node2(zone)
           A(val,node) = DOT_PRODUCT( foo(:), C(:,val,node) )
           B(val,node) = DOT_PRODUCT( foo(:), D(:,val,node) )
         enddo
       enddo
!$omp end target teams distribute parallel do

! Check that indirection in both loops is fine.  It may occur that different
! zones have different numbers of field values to solve.
!$omp target teams distribute parallel do collapse(2) &
!$omp& default(none) shared(num_field_values_arr,node1,node2,zone,A,B,C,D,foo,num_field_values) private(node, val)
       do val=1, num_field_values_arr(zone)
         do node= node1(zone), node2(zone)
           A(val,node) = DOT_PRODUCT( foo(:), C(:,val,node) )
           B(val,node) = DOT_PRODUCT( foo(:), D(:,val,node) )
         enddo
       enddo
!$omp end target teams distribute parallel do
     enddo
end program
