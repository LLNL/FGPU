program test
   implicit none
   
   integer, allocatable, dimension(:) :: node1
   integer, allocatable, dimension(:) :: node2
   integer, allocatable, dimension(:) :: faces_per_node

   real, allocatable, dimension(:,:) :: A
   real, allocatable, dimension(:,:) :: B
   real, allocatable, dimension(:,:,:) :: C
   real, allocatable, dimension(:,:,:) :: D
   real, allocatable, dimension(:) :: foo

   integer :: node, face, zone
   integer :: num_zones
   integer :: num_nodes_zone1
   integer :: num_nodes_zone2
   integer :: num_nodes_total
   integer :: num_dimensions
   integer :: max_faces_per_node

! NOTE - In this code example we have a case with two 3d zones.
! - One with 8 nodes (hex)
! - One with with 5 nodes (pyramid)
! - Each node is connected to a certain number of zone 'faces'.
! - The nodes in the hex zone connect to three faces.
! - The nodes at the bottom of the pyramid zone connects to three faces.
! - The node at the top of the pyramid zone connects to four faces.
! This example was pulled from an application code that supports 2d and 3d
! zones of arbitrary polygons and polyhedrals.  Zone shapes are not known
! until runtime, so a lookup is needed to determine iteration bounds in the loops.


   num_zones = 2
   num_nodes_zone1 = 8
   num_nodes_zone2 = 5
   num_nodes_total = num_nodes_zone1 + num_nodes_zone2
   num_dimensions = 3
   max_faces_per_node = 4

   ! Arrays holding indicies to first and last corner for each zone.
   allocate(node1(num_zones))
   node1(1) = 1
   node1(2) = num_nodes_zone1 + 1

   allocate(node2(num_zones))
   node2(1) = num_nodes_zone1
   node2(2) = num_nodes_zone1 + num_nodes_zone2

   allocate(faces_per_node( num_nodes_total ))
   faces_per_node(:) = 3
   ! Assume the first node of the pyramid is the top node, it has five faces.
   faces_per_node(9) = 4

   allocate(A(max_faces_per_node, num_nodes_total))
   allocate(B(max_faces_per_node, num_nodes_total))
   allocate(C(num_dimensions,max_faces_per_node, num_nodes_total))
   allocate(D(num_dimensions,max_faces_per_node, num_nodes_total))
   allocate(foo(num_dimensions))

   ! Some dummy data
   A = 1.0
   B = 2.0
   C = 3.0
   D = 4.0
   foo = 5.0

   print *, SHAPE(node1), " node1: ", node1
   print *, SHAPE(node2), " node2: ", node2
   print *, SHAPE(faces_per_node), " faces_per_node: ", faces_per_node

   do zone= 1, num_zones
       print *, "Zone: ", zone
!$omp parallel do collapse(2) default(none) shared(A,B,C,D,foo,faces_per_node) private(node, face)
       do node= node1(zone), node2(zone)
         do face=1, faces_per_node(node)
           A(face,node) = DOT_PRODUCT( foo(:), C(:,face,node) )
           B(face,node) = DOT_PRODUCT( foo(:), D(:,face,node) )
         enddo
       enddo
!$omp end parallel do

     enddo
end program
