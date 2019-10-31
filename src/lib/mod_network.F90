module mod_network

  use mod_kinds, only: ik, rk
  use mod_layer, only: array1d, array2d, db_init, dw_init,&
                       db_co_sum, dw_co_sum, layer_type
  use mod_dense_layer, only: Dense
  use mod_dropout_layer, only: Dropout
  use mod_batchnorm_layer, only: BatchNorm
  use mod_parallel, only: tile_indices

  implicit none

  private
  public :: network_type

  type layer_container
    class(layer_type), pointer :: p
  end type

  type :: network_type
    type(layer_container), allocatable :: layers(:)
    ! type(layer_type), allocatable :: layers(:)
    integer(ik) :: num_dense_layers
    real(rk), allocatable :: layer_info(:)
    character(len=100), allocatable :: layer_names(:)

  contains

    procedure, public, pass(self) :: accuracy
    procedure, public, pass(self) :: backprop
    procedure, public, pass(self) :: fwdprop
    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: loss
    procedure, public, pass(self) :: output
    procedure, public, pass(self) :: save
    ! procedure, public, pass(self) :: set_activation
    procedure, public, pass(self) :: sync
    procedure, public, pass(self) :: train_batch
    procedure, public, pass(self) :: train_single
    procedure, public, pass(self) :: update

    generic, public :: train => train_batch, train_single

  end type network_type

  interface network_type
    module procedure :: net_constructor
  endinterface network_type

contains

  type(network_type) function net_constructor(layer_names, layer_info) result(net)
    ! Network class constructor. Size of input array dims indicates the total
    ! number of layers (input + hidden + output), and the value of its elements
    ! corresponds the size of each layer.
    real(rk), intent(in) :: layer_info(:)
    character(len=100), intent(in) :: layer_names(:)

    call net % init(layer_names, layer_info)

    call net % sync(1)

  end function net_constructor

  real(rk) function accuracy(self, x, y)
    ! Given input x and output y, evaluates the position of the
    ! maximum value of the output and returns the number of matches
    ! relative to the size of the dataset.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:,:), y(:,:)
    integer(ik) :: i, good
    good = 0
    do i = 1, size(x, dim=2)
      if (all(maxloc(self % output(x(:,i))) == maxloc(y(:,i)))) then
        good = good + 1
      end if
    end do
    accuracy = real(good) / size(x, dim=2)
  end function accuracy

  pure subroutine backprop(self, y, dw, db)
    ! Applies a backward propagation through the network
    ! and returns the weight and bias gradients.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: y(:)
    type(array2d), allocatable, intent(out) :: dw(:)
    type(array1d), allocatable, intent(out) :: db(:)
    integer :: n, nm
    ! TODO: update
    ! associate(dims => self % dims, layers => self % layers)
    !
    !   call db_init(db, dims)
    !   call dw_init(dw, dims)
    !
    !   n = size(dims)
    !   db(n) % array = (layers(n) % a - y) * self % layers(n) % activation_prime(layers(n) % z)
    !   dw(n-1) % array = matmul(reshape(layers(n-1) % a, [dims(n-1), 1]),&
    !                            reshape(db(n) % array, [1, dims(n)]))
    !
    !   do n = size(dims) - 1, 2, -1
    !     db(n) % array = matmul(layers(n) % w, db(n+1) % array)&
    !                   * self % layers(n) % activation_prime(layers(n) % z)
    !     dw(n-1) % array = matmul(reshape(layers(n-1) % a, [dims(n-1), 1]),&
    !                              reshape(db(n) % array, [1, dims(n)]))
    !   end do
    !
    ! end associate

  end subroutine backprop

  subroutine fwdprop(self, input)
    ! Performs the forward propagation and stores arguments to activation
    ! functions and activations themselves for use in backprop.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: input(:)
    integer(ik) :: n

    ! forward through first layer
    call self % layers(1) % p % forward(input)

    ! iterate through rest of the layers
    do n = 2, size(self % layers) - 1
      call self % layers(n) % p % forward(self % layers(n-1) % p % o)
    end do

  end subroutine fwdprop

  subroutine init(self, layer_names, layer_info)
    ! Allocates and initializes the layers with given dimensions dims.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: layer_info(:)
    integer(ik), allocatable :: dense_dims(:)
    integer(ik) :: n, i, dense_idx, unique_layers
    character(len=100), intent(in) :: layer_names(:)

    dense_idx = 0
    unique_layers = 0
    allocate(dense_dims(size(layer_info)))

    do n = 1, size(layer_names)
      ! count unique layers (i.e. not activations)
      select case(trim(layer_names(n)))
        case('input')
          dense_idx = dense_idx + 1
          unique_layers = unique_layers + 1
          ! store the dimensions of the weights
          dense_dims(dense_idx) = layer_info(n)
        case('dense')
          dense_idx = dense_idx + 1
          unique_layers = unique_layers + 1
          ! store the dimensions of the weights
          dense_dims(dense_idx) = layer_info(n)
        case('dropout')
          unique_layers = unique_layers + 1
        case('batchnorm')
          unique_layers = unique_layers + 1
      end select
    end do

    ! allocate the number of unique layers (i.e. not activations)
    if (.not. allocated(self % layers)) allocate(self % layers(unique_layers))

    dense_idx = 0
    unique_layers = 0

    do n = 1, size(layer_names)
      ! self % layers(n) = layer_type(dims(n), dims(n+1))
      select case(trim(layer_names(n)))
        case('input')
          dense_idx = dense_idx + 1
          unique_layers = unique_layers + 1

          allocate(&
            self % layers(unique_layers) % p,&
            source=Dense(dense_dims(dense_idx), dense_dims(dense_idx + 1),&       ! shape of dense layer
              'linear', 0.0)&                                                     ! input layer always has linear activation
          )
        case('dense')
          dense_idx = dense_idx + 1
          unique_layers = unique_layers + 1

          allocate(&
            self % layers(unique_layers) % p,&
            source=Dense(&
              dense_dims(dense_idx), dense_dims(dense_idx + 1),&                  ! shape of dense layer
              layer_names(n + 1), layer_info(n + 1))&                                     ! activation function args
          )
        case('dropout')
          unique_layers = unique_layers + 1

          allocate(&
            self % layers(unique_layers) % p,&
            source=Dropout(dense_dims(dense_idx), layer_info(n))&
          )
        case('batchnorm')
          unique_layers = unique_layers + 1

          ! allocate(&
          !   self % layers(unique_layers) % p,&
          !   source=BatchNorm(dense_dims(dense_idx), layer_info(n))&
          ! )
      end select

    end do

    self % layer_info = layer_info
    self % layer_names = layer_names
    self % num_dense_layers = dense_idx

  end subroutine init

  subroutine load(self, filename)
    ! Loads the network from file.
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    ! character(len=20) :: activation_type
    integer(ik) :: fileunit, n, num_layers
    character(len=100), allocatable :: layer_names(:)
    real(rk), allocatable :: layer_info(:)

    open(newunit=fileunit, file=filename, status='old', action='read')

    ! number of layers in network; this includes input and activations
    read(fileunit, fmt=*) num_layers

    ! allocate storage
    allocate(layer_names(num_layers))
    allocate(layer_info(num_layers))

    ! read through the network description
    do n = 1, num_layers
      read(fileunit, fmt=*) layer_names(n), layer_info(n)
    end do

    ! initialize the network
    call self % init(layer_names, layer_info)

    ! Read in weights and biases
    ! input layer doesn't have biases
    do n = 2, size(self % layers)
      select type (layer => self % layers(n) % p)
        class is (Dense)
          read(fileunit, fmt=*) self % layers(n) % p % b
      end select
    end do

    ! read weights into layers
    do n = 1, size(self % layers) - 1
      select type (layer => self % layers(n) % p)
        class is (Dense)
          read(fileunit, fmt=*) self % layers(n) % p % w
      end select
    end do

    close(fileunit)

  end subroutine load

  real(rk) function loss(self, x, y)
    ! Given input x and expected output y, returns the loss of the network.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:), y(:)
    loss = 0.5 * sum((y - self % output(x))**2) / size(x)
  end function loss

  function output(self, input) result(a)
    ! Use forward propagation to compute the output of the network.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: input(:)
    real(rk), allocatable :: a(:)
    integer(ik) :: n

    associate(layers => self % layers)
      ! pass input to first layer
      call layers(1) % p % forward(input)

      ! iterate through layers passing activation forward
      do n = 2, size(layers) - 1
        call layers(n) % p % forward(layers(n-1) % p % o)
      end do

      ! get activation from last layer
      a = layers(size(layers) - 1) % p % o
    end associate

  end function output


  subroutine save(self, filename)
    ! Saves the network to a file.
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer(ik) :: fileunit, n

    open(newunit=fileunit, file=filename)
    ! total number of operations including activations
    write(fileunit, fmt=*) size(self % layer_info)

    do n = 1, size(self % layer_names)
      ! layer name \t info
      write(fileunit, fmt=*) self % layer_names(n), self % layer_info(n)
    end do

    do n = 2, size(self % layers)
      select type (layer => self % layers(n) % p)
        class is (Dense)
          ! write biases of dense layer
          write(fileunit, fmt=*) self % layers(n) % p % b
      end select
    end do

    do n = 1, size(self % layers) - 1
      select type (layer => self % layers(n) % p)
        class is (Dense)
          ! write weights of dense layer
          write(fileunit, fmt=*) self % layers(n) % p % w
      end select
    end do

    close(fileunit)

  end subroutine save

  ! pure subroutine set_activation(self, activation)
  !   ! A thin wrapper around layer % set_activation().
  !   ! This method can be used to set an activation function
  !   ! for all layers at once.
  !   class(network_type), intent(in out) :: self
  !   character(len=*), intent(in) :: activation
  !   integer :: n
  !   do concurrent(n = 1:size(self % layers))
  !     call self % layers(n) % p % set_activation(activation)
  !   end do
  ! end subroutine set_activation

  subroutine sync(self, image)
    ! Broadcasts network weights and biases from
    ! specified image to all others.
    class(network_type), intent(in out) :: self
    integer(ik), intent(in) :: image
    integer(ik) :: n
    if (num_images() == 1) return
    layers: do n = 1, size(self % layers) ! changed from dims
#ifdef CAF
      call co_broadcast(self % layers(n) % p % b, image)
      call co_broadcast(self % layers(n) % p % w, image)
#endif
    end do layers
  end subroutine sync

  subroutine train_batch(self, x, y, eta)
    ! Trains a network using input data x and output data y,
    ! and learning rate eta. The learning rate is normalized
    ! with the size of the data batch.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:,:), y(:,:), eta
    type(array1d), allocatable :: db(:), db_batch(:)
    type(array2d), allocatable :: dw(:), dw_batch(:)
    ! TODO: update
    ! integer(ik) :: i, im, n, nm
    ! integer(ik) :: is, ie, indices(2)
    !
    ! im = size(x, dim=2) ! mini-batch size
    ! nm = size(self % dims) ! number of layers
    !
    ! ! get start and end index for mini-batch
    ! indices = tile_indices(im)
    ! is = indices(1)
    ! ie = indices(2)
    !
    ! call db_init(db_batch, self % dims)
    ! call dw_init(dw_batch, self % dims)
    !
    ! do concurrent(i = is:ie)
    !   call self % fwdprop(x(:,i))
    !   call self % backprop(y(:,i), dw, db)
    !   do concurrent(n = 1:nm)
    !     dw_batch(n) % array =  dw_batch(n) % array + dw(n) % array
    !     db_batch(n) % array =  db_batch(n) % array + db(n) % array
    !   end do
    ! end do
    !
    ! if (num_images() > 1) then
    !   call dw_co_sum(dw_batch)
    !   call db_co_sum(db_batch)
    ! end if
    !
    ! call self % update(dw_batch, db_batch, eta / im)

  end subroutine train_batch

  subroutine train_single(self, x, y, eta)
    ! Trains a network using a single set of input data x and output data y,
    ! and learning rate eta.
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:), y(:), eta
    type(array2d), allocatable :: dw(:)
    type(array1d), allocatable :: db(:)
    call self % fwdprop(x)
    call self % backprop(y, dw, db)
    call self % update(dw, db, eta)
  end subroutine train_single

  pure subroutine update(self, dw, db, eta)
    ! Updates network weights and biases with gradients dw and db,
    ! scaled by learning rate eta.
    class(network_type), intent(in out) :: self
    class(array2d), intent(in) :: dw(:)
    class(array1d), intent(in) :: db(:)
    real(rk), intent(in) :: eta
    integer(ik) :: n
    ! TODO: update
    ! associate(layers => self % layers, nm => size(self % dims))
    !   ! update biases
    !   do concurrent(n = 2:nm)
    !     layers(n) % b = layers(n) % b - eta * db(n) % array
    !   end do
    !   ! update weights
    !   do concurrent(n = 1:nm-1)
    !     layers(n) % w = layers(n) % w - eta * dw(n) % array
    !   end do
    ! end associate

  end subroutine update

end module mod_network
