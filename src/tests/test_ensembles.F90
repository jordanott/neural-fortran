! test_ensembles.f90

! load ensemble members in directory
! get prediction
!   each model in the ensemble makes a prediction
!   results are averaged over all members

program test_ensembles
  use mod_kinds, only: ik, rk
  use mod_ensemble, only: ensemble_type

  implicit none

  type(ensemble_type) :: ensemble

  real(rk), allocatable :: result1(:), input(:)
  character(len=100), dimension(:), allocatable :: args

  allocate(args(1))
  call get_command_argument(1,args(1))

  ! build ensemble from members in specified directory
  ensemble = ensemble_type(args(1))

  input = [5, 4, 5, 9, 7, 5, 0, 3, 9, 4, 8, 0, 4, 5, 5, 4, 1, 0, 0, 4, 2, 8,&
    2, 1, 7, 1, 6, 7, 4, 1, 4, 2, 6, 1, 9, 1, 7, 8, 7, 5, 8, 6, 3, 6,&
    4, 7, 0, 5, 4, 1, 0, 1, 9, 6, 7, 3, 0, 3, 4, 1, 6, 2, 4, 1, 3, 7,&
    6, 7, 8, 6, 7, 4, 5, 8, 8, 6, 0, 6, 9, 2, 5, 4, 1, 6, 9, 8, 7, 8,&
    5, 1, 2, 1, 1, 6]

  ! run test input through network
  result1 = ensemble % average(input)
  print *, result1

end program test_ensembles
