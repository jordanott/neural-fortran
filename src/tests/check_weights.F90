program check_weights
  use mod_network!, only: network_type
  implicit none
  type(network_type) :: net
  integer :: n

  call net % load('/home1/06528/tg858273/saved_models/model.txt')

  do n = 1, size(net % layers)
    print *, net % layers(n) % w
  end do



end program check_weights
