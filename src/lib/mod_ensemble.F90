module mod_ensemble

  use mod_kinds, only: ik, rk
  use mod_random, only: randn
  use mod_network

  implicit none

  private
  public :: ensemble_type

  type network_container
    class(network_type), pointer :: p
  end type

  type :: ensemble_type
    type(network_container), allocatable :: ensemble_members(:)
    integer(ik) :: num_members, num_of_each, total_members

  contains

    procedure, public, pass(self) :: average

  end type ensemble_type

  interface ensemble_type
    module procedure :: ensemble_constructor
  endinterface ensemble_type

contains

  type(ensemble_type) function ensemble_constructor(directory) result(ensemble)
    ! creates a network for every config txt in the directory
    real :: r
    integer :: i,n,reason
    character(len=*), intent(in) :: directory
    character(LEN=100), dimension(:), allocatable :: model_file_names
    type(network_type) :: net

    ! get the files
    call system('ls '//directory//' > ensemble_members.txt')
    open(31,FILE='ensemble_members.txt',action="read")

    ! count how many files in directory
    ensemble % num_members = 0
    do
      read(31,FMT='(a)',iostat=reason) r
      if (reason/=0) EXIT
      ensemble % num_members = ensemble % num_members + 1
    end do

    ! allocate how many members in the ensemble
    allocate(ensemble % ensemble_members(ensemble % num_members))
    allocate(model_file_names(ensemble % num_members))
    rewind(31)

    do i = 1,ensemble % num_members
      read(31,'(a)') model_file_names(i)
      ! print *, trim(directory)//trim(model_file_names(i))
      ! construct model using the config file
      call net % load(trim(directory)//trim(model_file_names(i)))

      allocate(&
        ensemble % ensemble_members(i) % p,&
        source=net&
      )

    end do

    close(31)

    ensemble % num_of_each = int(128 / ensemble % num_members)
    ensemble % total_members = ensemble % num_members * ensemble % num_of_each

  end function ensemble_constructor


  function average(self, input) result(output)
    ! Use forward propagation to compute the output of the network.
    class(ensemble_type), intent(in out) :: self
    real(rk), intent(in) :: input(:)
    real(rk), allocatable :: output(:)
    integer(ik) :: i,j, output_size, input_size

    ! getting output size from model
    input_size  = self % ensemble_members(1) % p % input_size
    output_size = self % ensemble_members(1) % p % output_size
    allocate(output(output_size))

    associate(members => self % ensemble_members)
      do i=1, self % num_members
        do j=1, self % num_of_each
          ! output from model - noise added to input
          output = output + members(i) % p % output(&
            input + randn(input_size)* 0.01&
          )
        end do
      end do
    end associate
    ! average over all model predictions
    output = output / self % total_members

  end function average


end module mod_ensemble
