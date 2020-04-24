module objects
  use prim

  type(prim_type),    target  ::    prim_data(1:10)
  type(prim_type),    pointer ::    prim_ptr  

  contains

    subroutine setup_objects(id)
      implicit none
      integer, intent(in) :: id

      call prim_data(id)%setup()

    end subroutine setup_objects

    subroutine point_to_objects(id)
      implicit none
      integer(c_int), intent(in) :: id
      prim_ptr   =>   prim_data(id)
    end subroutine point_to_objects


end module objects
