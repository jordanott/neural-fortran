module mod_batchnorm_layer

  use mod_layer
  use mod_kinds, only: ik, rk

  implicit none

  ! BatchNorm layer - extends from base layer_type
  !   Implements the dropout algorithm
  type, extends(layer_type) :: BatchNorm
    ! probability of dropping a node
    real(rk) :: drop_prob

  contains

    procedure, public, pass(self) :: forward => dense_forward
    procedure, public, pass(self) :: backward => dense_backward

  end type BatchNorm

  interface BatchNorm
    module procedure :: constructor
  end interface BatchNorm

contains

  type(BatchNorm) function constructor(this_size, drop_prob) result(layer)
    ! BatchNorm class constructor
    !   this_size: size to allocate for current layer
    !   drop_prob: probability of dropping a node

    integer(ik), intent(in) :: this_size
    real(rk), intent(in) :: drop_prob
    allocate(layer % o(this_size))

    ! store layer drop probability
    layer % drop_prob = drop_prob
    ! not in training mode
    layer % training = .FALSE.

    ! print *, 'Creating dropout layer', this_size, drop_prob
  end function constructor


  subroutine dense_forward(self, x)

    class(BatchNorm), intent(in out) :: self
    real(rk), intent(in) :: x(:)

    if (self % training) then
      ! TODO:
      self % o = x * self % drop_prob
    else
      ! NOT TRAINING: pass output forward
      self % o = x
    end if

  end subroutine dense_forward


  subroutine dense_backward(self, x)

    class(BatchNorm), intent(in out) :: self
    real(rk), intent(in) :: x(:)

    ! TODO: implement backward pass
  end subroutine dense_backward

end module mod_batchnorm_layer
