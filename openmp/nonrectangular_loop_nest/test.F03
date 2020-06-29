program test
   implicit none
   
   integer, allocatable, dimension(:) :: corner1
   integer, allocatable, dimension(:) :: corner2
   integer, allocatable, dimension(:) :: corner_faces_per_corner

   real, allocatable, dimension(:,:) :: A
   real, allocatable, dimension(:,:) :: B
   real, allocatable, dimension(:,:,:) :: C
   real, allocatable, dimension(:,:,:) :: D
   real, allocatable, dimension(:) :: foo

   integer :: corner, cornerface, zone

! NOTE - In this code example we have just a trivial case with one 3d zone with 8 corners, and 3
! 'corner faces' per corner.  This example was pulled from an application code
! that supported 2d and 3d zones of arbitrary polygons and polyhedrals.
! It could not be assumed that each zone had 8 corners and 3 corner faces per
! corner, so a lookup was required in the loops to get the correct number of
! corners or corner faces to loop over.
!
   integer, parameter :: num_zones = 1
   integer, parameter :: num_corners_per_zone = 8
   integer, parameter :: num_corner_faces_per_corner = 3
   integer, parameter :: num_dimensions = 3

   ! Arrays holding indicies to first and last corner for each zone.
   allocate(corner1(num_zones))
   corner1(1) = 1

   allocate(corner2(num_zones))
   corner2(1) = 8

   allocate(corner_faces_per_corner(num_zones*num_corners_per_zone))
   corner_faces_per_corner=3

   allocate(A(num_corner_faces_per_corner, num_corners_per_zone))
   allocate(B(num_corner_faces_per_corner, num_corners_per_zone))
   allocate(C(num_dimensions,num_corner_faces_per_corner, num_corners_per_zone))
   allocate(D(num_dimensions,num_corner_faces_per_corner, num_corners_per_zone))
   allocate(foo(num_corner_faces_per_corner))

   ! Some dummy data
   A = 1.0
   B = 2.0
   C = 3.0
   D = 4.0
   foo = 5.0

   do zone= 1, num_zones
     
!$omp parallel do collapse(2) default(none) shared(A,B,C,D,foo,corner_faces_per_corner) private(cornerface)
       do corner= corner1(zone), corner2(zone)
         do cornerface=1, corner_faces_per_corner(corner)
           A(cornerface,corner) = DOT_PRODUCT( foo(:), C(:,cornerface,corner) )
           B(cornerface,corner) = DOT_PRODUCT( foo(:), D(:,cornerface,corner) )
         enddo
       enddo
!$omp end parallel do

     enddo
end program
