! test_keras.f90

! read in some data
! read in some network from .txt(?) file
!   network will have been trained in keras
! run data through network
! check that result is same as in keras


program test_keras

  ! use mod_mnist, only: load_mnist
  ! use mod_kinds, only: ik, rk
  use mod_kinds, only: ik, rk
  use mod_mnist, only: label_digits, load_mnist
  use mod_network, only: network_type

  implicit none

  real(rk), allocatable :: tr_images(:,:), tr_labels(:)
  real(rk), allocatable :: te_images(:,:), te_labels(:)
  real(rk), allocatable :: va_images(:,:), va_labels(:)
  ! real(rk), allocatable :: input(:,:), output(:,:)

  type(network_type) :: net

  real(rk), allocatable :: result1(:)

  integer(ik) :: fileunit, num_layers !i, n, num_epochs, num_layers,fileunit
  ! integer(ik) :: batch_size, batch_start, batch_end
  ! real(rk) :: pos
  ! real(rk), allocatable :: input(:)

  call load_mnist(tr_images, tr_labels, te_images, te_labels, va_images, va_labels)
  print*, 'tr_images shape'
  print*, shape(tr_images)
  print*, 'first training image'
  print*,size(tr_images(:,1))
  ! print*,tr_images(:,1)
  ! write it to a file to try in python/keras
  open (unit = 1, file = 'testcase.txt')
  write(1,*) tr_images(:,1)

  ! load trained network from keras
  call net % load('../../src/new_keras10.txt')
  print *, 'loaded new_keras10.txt'

  ! num_layers=size(net % dims)
  ! print *, 'dims'
  ! print *, net%dims(num_layers)

  ! run trail image through network
  result1 = net % output(tr_images(:,1))
  print *, result1

end program test_keras
