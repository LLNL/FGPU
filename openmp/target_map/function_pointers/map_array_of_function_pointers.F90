program Main

	implicit none
	abstract interface
	
		function func (z)
			!$OMP declare target
			integer :: func
			integer, intent (in) :: z
		end function func
		
	end interface
	integer,parameter::scaler=2

	call Driver
	
	contains
	!############################################################################
	subroutine Driver

		implicit none
		procedure(func),pointer:: fun_pointer
		integer::counter
		integer::vector(10)
		integer:: shift
		
		shift=1

		!$OMP target teams distribute parallel do
		do counter=1,10
			fun_pointer=>Fun_of_x
			vector(counter)=Eval_Fun(fun_pointer,counter,shift)
		enddo
		!$OMP end target teams distribute parallel do

		do counter=1,10
			write(*,*)'vector(counter)=',vector(counter)
            if (vector(counter).NE.(counter*2-1)) then
                write(*,*) 'wrong value'
                STOP 112
            endif
        enddo
	
	end subroutine Driver
	!############################################################################
	function Fun_of_x(x)

		implicit none
		!$OMP declare target
		integer, intent(in)::x
		integer::Fun_of_x
		
		Fun_of_x=scaler*x
		
	end function Fun_of_x
	!############################################################################
	function Eval_Fun(fun_pointer,x,shift)
	
		implicit none
		procedure(func),pointer,intent(in)::fun_pointer
		integer,intent(in)::x,shift
		integer::Eval_Fun
		
		!$OMP target data map(tofrom: Eval_fun)
			Eval_Fun=fun_pointer(x)-shift
		!$OMP end target data

	end function Eval_Fun
	!############################################################################


end program Main
