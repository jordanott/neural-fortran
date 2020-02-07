module mod_ensemble

  use omp_lib
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
    type(network_type) :: model
    type(network_container), allocatable :: ensemble_members(:)
    real(rk) :: noise
    integer(ik) :: num_members, num_of_each, total_members
  contains

    procedure, public, pass(self) :: average

  end type ensemble_type

  interface ensemble_type
    module procedure :: ensemble_constructor
  endinterface ensemble_type

contains

  ! ------------------- Multiple Ensemble Members ---------------------
  !
  type(ensemble_type) function ensemble_constructor(directory, noise) result(ensemble)
    ! creates a network for every config txt in the directory
    real :: r
    integer :: i,n,reason, idx
    real(rk), intent(in) :: noise
    character(len=*), intent(in) :: directory
    character(LEN=100), dimension(:), allocatable :: model_file_names
    type(network_type) :: net

    ! get the files
    call system('ls '//directory//' > ensemble_members.txt')
    open(31,FILE='ensemble_members.txt',action="read")

    ! set standard deviation for noise perturbation
    ensemble % noise = noise
    ! count how many files in directory
    ensemble % num_members = 0
    do
      read(31,FMT='(a)',iostat=reason) r
      if (reason/=0) EXIT
      ensemble % num_members = ensemble % num_members + 1
    end do

    ensemble % num_of_each = int(128 / ensemble % num_members)
    ensemble % total_members = ensemble % num_members * ensemble % num_of_each

    ! allocate how many members in the ensemble
    allocate(ensemble % ensemble_members(ensemble % total_members))
    allocate(model_file_names(ensemble % total_members))
    rewind(31)

    do i = 1,ensemble % num_members
      read(31,'(a)') model_file_names(i)

      ! construct model using the config file
      call net % load(trim(directory)//trim(model_file_names(i)))

      do n = 1, ensemble % num_of_each
        idx = (i - 1) * ensemble % num_of_each + n

        allocate(&
          ensemble % ensemble_members(idx) % p,&
          source=net&
        )
      end do
    end do

    close(31)

  end function ensemble_constructor


  function average(self, input) result(output)
    ! Use forward propagation to compute the output of the network.
    class(ensemble_type), intent(in out) :: self
    real(rk), intent(in) :: input(:)
    real(rk), allocatable :: storage(:,:), model_output(:), output(:)
    integer(ik) :: i,j,idx, output_size, input_size

    ! getting output size from model
    input_size  = self % ensemble_members(1) % p % input_size
    output_size = self % ensemble_members(1) % p % output_size
    ! allocate a space for each model to write in
    allocate(storage(output_size, self % total_members))
    allocate(output(output_size))
    allocate(model_output(output_size))

    !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,output)
    do i=1, self % total_members
      ! just to check we're using multiple threads
      ! print *, OMP_GET_THREAD_NUM(), i

      ! output from model - noise added to input
      model_output = self % ensemble_members(i) % p % output(&
        input + randn(input_size) * self % noise&
      )

      ! write model output into shared memory
      do j=1, output_size
        storage(j, i) = model_output(j)
      end do

    end do
    !$OMP END PARALLEL DO

    ! average over all model predictions
    output = sum(storage, DIM=2) / self % total_members

  end function average


end module mod_ensemble


! OLD CODE FOR SINGLE ENSEMBLE MEMBER
! type(ensemble_type) function ensemble_constructor(directory) result(ensemble)
!   ! creates a network for every config txt in the directory
!   character(len=*), intent(in) :: directory
!
!   ensemble % total_members = 128
!
!   call ensemble % model % load(directory)
! end function ensemble_constructor
!
!
! function average(self, input) result(output)
!   ! Use forward propagation to compute the output of the network.
!   class(ensemble_type), intent(in out) :: self
!   real(rk), intent(in) :: input(:)
!   real(rk), allocatable :: storage(:,:), model_output(:), output(:)
!   integer(ik) :: i,j,idx, output_size, input_size
!
!   ! getting output size from model
!   input_size  = self % model % input_size
!   output_size = self % model % output_size
!   ! allocate a space for each model to write in
!   allocate(storage(output_size, self % total_members))
!   allocate(output(output_size))
!   allocate(model_output(output_size))
!
!   !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,output)
!   do i=1, self % total_members
!     ! just to check we're using multiple threads
!     idx = OMP_GET_THREAD_NUM()
!     if ( idx /= 0 ) then
!       print *, idx
!     end if
!
!     ! output from model - noise added to input
!     model_output = self % model % output(&
!       input + randn(input_size)* 0.01&
!     )
!
!     ! write model output into shared memory
!     do j=1, output_size
!       storage(j, i) = model_output(j)
!     end do
!
!   end do
!   !$OMP END PARALLEL DO
!
!   ! average over all model predictions
!   output = sum(storage, DIM=2) / self % total_members
!
! end function average
