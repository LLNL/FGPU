module ompdata

  use objects
  implicit none


  contains

    subroutine ompdata_prim_to()
      implicit none
      !$omp target enter data map(to:prim_ptr%v1,prim_ptr%v2,prim_ptr%v3)
    end subroutine ompdata_prim_to

    subroutine ompdata_prim_from()
      implicit none
      !$omp target exit data map(from:prim_ptr%v1,prim_ptr%v2,prim_ptr%v3)
    end subroutine ompdata_prim_from
    
    subroutine ompdata_prim_delete()
      implicit none
      !$omp target exit data map(delete:prim_ptr%v1,prim_ptr%v2,prim_ptr%v3)
    end subroutine ompdata_prim_delete        

  
end module ompdata
