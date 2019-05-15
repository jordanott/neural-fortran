! program test_network_save
!   use mod_network, only: network_type
!   implicit none
!   type(network_type) :: net1, net2
!   integer :: n
!   print *, 'Initializing 2 networks with random weights and biases'
!   net1 = network_type([768, 30, 10])
!   net2 = network_type([768, 30, 10])
!   print *, 'Save network 1 into file'
!   call net1 % save('test_network.txt')
!   call net2 % load('test_network.txt')
!   print *, 'Load network 2 from file'
!   do n = 1, size(net1 % layers)
!     print *, 'Layer ', n, ', weights equal: ',&
!       all(net1 % layers(n) % w == net2 % layers(n) % w),&
!       ', biases equal:', all(net1 % layers(n) % b == net2 % layers(n) % b)
!   end do
! end program test_network_save
! test load and Accuracy
program test_network_save
  use mod_network!, only: network_type
  use mod_kinds, only: ik, rk
  use mod_mnist, only: label_digits, load_mnist

  implicit none
  type(network_type) :: net1, net2
  integer :: n

  real(rk), allocatable :: tr_images(:,:), tr_labels(:)
  real(rk), allocatable :: te_images(:,:), te_labels(:)
  !real(rk), allocatable :: va_images(:,:), va_labels(:)
  real(rk), allocatable :: input(:,:), output(:,:)
  call load_mnist(tr_images, tr_labels, te_images, te_labels)


  print *, 'Initializing 2 networks with random weights and biases'
  net1 = network_type([784, 30, 10])
  net2 = network_type([784, 30, 10])


  print *, 'Save network 1 into file'
  call net1 % save('test_network.txt')
  print *, 'Load network 2 from file'
  call net2 % load('test_network.txt')


  print *, 'Checking weights and biases'
  do n = 1, size(net1 % layers)
    print *, 'Layer ', n, ', weights equal: ',&
      all(net1 % layers(n) % w == net2 % layers(n) % w),&
      ', biases equal:', all(net1 % layers(n) % b == net2 % layers(n) % b)
  end do


  print *, 'Attempt accuracy with loaded network'
  if (this_image() == 1) then
     write(*, '(a,f5.2,a)') 'Initial accuracy: ',&
       net2 % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
  end if
  
end program test_network_save
