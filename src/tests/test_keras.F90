! test_keras.f90

! read in some data
! read in some network from .txt(?) file
!   network will have been trained in keras
! run data through network
! check that result is same as in keras

program test_keras
  use mod_kinds, only: ik, rk
  use mod_network, only: network_type

  implicit none

  type(network_type) :: net

  real(rk), allocatable :: result1(:), input(:)
  character(len=100), dimension(:), allocatable :: args

  allocate(args(1))
  call get_command_argument(1,args(1))

  ! load trained network from keras
  call net % load(args(1))

  input = [1, 2, 3, 4, 5]

  ! run test input through network
  result1 = net % output(input)
  print *, result1

end program test_keras
